"""
Backtest: 6 Strategy Variants from Pairwise Ranker v2
======================================================
Uses the trained pairwise ranking model to run proper backtests:
  1. Long-Only Q10 (top 10%)
  2. Short-Only Q1 (bottom 10%)
  3. Long-Short Q10/Q1 (10/10)
  4. Long-Only Q20 (top 20%)
  5. Short-Only Q20 (bottom 20%)
  6. Long-Short Q20/Q20 (20/20)

Plus: BTC Buy&Hold benchmark and CatBoost Short-Only baseline comparison.

Costs modeled: maker+slippage per side, funding on shorts.
"""
import numpy as np
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings('ignore')

from numba import njit
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reuse model infrastructure from v2
from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _train_model, _predict, _spearman_corr,
    _sample_pairs_and_train_epoch,
    TRAIN_DAYS, PURGE_DAYS, HOLDING_DAYS, REBAL_EVERY, MIN_COINS,
    VOL_FLOOR_PCT, N_EPOCHS, PAIRS_PER_DATE, NEAR_TIE_PCT,
    TAIL_WEIGHT_POW, LR, L1_REG, L2_REG, RETRAIN_EVERY,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ─── Cost Model (Binance) ───
COST_PER_SIDE = 0.0007      # maker 0.02% + slippage 0.05% = 0.07%
FUNDING_DAILY = 0.0003      # ~11% annualized funding cost for shorts
INITIAL_CAPITAL = 100_000

# ─── Strategy Definitions ───
STRATEGIES = {
    'long_q10':      {'long_pct': 0.10, 'short_pct': 0.00, 'label': 'Long Top 10%'},
    'short_q10':     {'long_pct': 0.00, 'short_pct': 0.10, 'label': 'Short Bot 10%'},
    'ls_q10':        {'long_pct': 0.10, 'short_pct': 0.10, 'label': 'L/S 10-10%'},
    'long_q20':      {'long_pct': 0.20, 'short_pct': 0.00, 'label': 'Long Top 20%'},
    'short_q20':     {'long_pct': 0.00, 'short_pct': 0.20, 'label': 'Short Bot 20%'},
    'ls_q20':        {'long_pct': 0.20, 'short_pct': 0.20, 'label': 'L/S 20-20%'},
}


def run_backtest():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    # Load data
    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    n_feat = len([f for f in feat_cols if f in panel.columns])

    fwd_col = f'fwd_ret_{HOLDING_DAYS}d'
    if fwd_col not in panel.columns:
        panel[fwd_col] = panel.groupby(level='symbol')['close'].transform(
            lambda x: x.pct_change(HOLDING_DAYS).shift(-HOLDING_DAYS)
        )

    # Build date groups
    print("Building date groups...")
    date_groups = {}
    for date, group in panel.groupby(level='date'):
        g = group.dropna(subset=[fwd_col, 'close'])
        if len(g) < MIN_COINS:
            continue
        if 'turnover_28d' in g.columns:
            thresh = g['turnover_28d'].quantile(VOL_FLOOR_PCT)
            g = g[g['turnover_28d'] >= thresh]
        if len(g) < MIN_COINS:
            continue

        available = [f for f in feat_cols if f in g.columns]
        features = g[available].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = _cross_sectional_rank_normalize(features)

        fwd_ret = g[fwd_col].values.astype(np.float64)
        fwd_ret = np.nan_to_num(fwd_ret, nan=0.0)
        market_ret = np.nanmean(fwd_ret)
        excess_ret = fwd_ret - market_ret

        # BTC return for benchmark
        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []
        btc_ret = 0.0
        if 'BTCUSDT' in syms:
            btc_idx = syms.index('BTCUSDT')
            btc_ret = fwd_ret[btc_idx]

        date_groups[date] = {
            'features': features,
            'fwd_ret': fwd_ret,
            'excess_ret': excess_ret,
            'symbols': syms,
            'btc_ret': btc_ret,
            'n_coins': len(syms),
        }

    sorted_dates = sorted(date_groups.keys())
    print(f"  {len(sorted_dates)} valid dates, {n_feat} features")

    # Walk-forward schedule
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if (d - sorted_dates[0]).days >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i
            break
    rebal_dates = sorted_dates[start_idx::REBAL_EVERY]
    print(f"  {len(rebal_dates)} rebalance dates")

    # JIT warmup
    dummy = np.random.randn(20, n_feat)
    dummy_e = np.random.randn(20)
    dummy_w = np.zeros(n_feat)
    _sample_pairs_and_train_epoch(dummy, dummy_e, dummy_w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # ═══ Walk-forward with backtest ═══
    print("Running walk-forward backtest...")
    t0 = time.time()

    # Equity curves per strategy
    equity = {name: INITIAL_CAPITAL for name in STRATEGIES}
    equity['btc_bh'] = INITIAL_CAPITAL
    curves = {name: [] for name in list(STRATEGIES.keys()) + ['btc_bh']}

    period_log = []
    weights = np.zeros(n_feat)
    last_train_ri = None

    for ri, pred_date in enumerate(rebal_dates):
        should_train = (last_train_ri is None or ri % RETRAIN_EVERY == 0)

        if should_train:
            train_end = pred_date - pd.Timedelta(days=PURGE_DAYS)
            train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
            train_dates = [d for d in sorted_dates
                          if train_start <= d <= train_end and d in date_groups]
            if len(train_dates) < 10:
                continue

            all_feat_list, all_eret_list, offsets = [], [], [0]
            for td in train_dates:
                dg = date_groups[td]
                all_feat_list.append(dg['features'])
                all_eret_list.append(dg['excess_ret'])
                offsets.append(offsets[-1] + len(dg['excess_ret']))

            weights = _train_model(
                np.vstack(all_feat_list), np.concatenate(all_eret_list),
                np.array(offsets, dtype=np.int64),
                n_feat, N_EPOCHS, PAIRS_PER_DATE,
                NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG,
                42 + ri,
            )
            last_train_ri = ri
            if ri == 0 or (ri + 1) % 20 == 0:
                print(f"  [{ri+1}/{len(rebal_dates)}] {pred_date.date()} [{time.time()-t0:.0f}s]",
                      flush=True)

        if last_train_ri is None:
            continue
        if pred_date not in date_groups:
            continue

        dg = date_groups[pred_date]
        scores = _predict(dg['features'], weights)
        n_coins = dg['n_coins']
        fwd_ret = dg['fwd_ret']
        sorted_idx = np.argsort(scores)
        date_str = str(pred_date.date())

        # Rank IC
        rank_ic = float(_spearman_corr(scores, dg['excess_ret']))

        # ─── Compute returns for each strategy ───
        period_entry = {'date': pred_date, 'date_str': date_str,
                        'n_coins': n_coins, 'rank_ic': rank_ic}

        for name, cfg in STRATEGIES.items():
            long_pct = cfg['long_pct']
            short_pct = cfg['short_pct']

            n_long = max(1, int(n_coins * long_pct)) if long_pct > 0 else 0
            n_short = max(1, int(n_coins * short_pct)) if short_pct > 0 else 0

            # Long: top scores, Short: bottom scores
            long_ret = 0.0
            if n_long > 0:
                long_idx = sorted_idx[-n_long:]
                long_ret = np.mean(fwd_ret[long_idx])

            short_ret = 0.0
            if n_short > 0:
                short_idx = sorted_idx[:n_short]
                short_ret = -np.mean(fwd_ret[short_idx])  # profit from shorting
                # Funding cost on shorts
                short_ret -= FUNDING_DAILY * HOLDING_DAYS

            # Combine
            if n_long > 0 and n_short > 0:
                # L/S: equal weight each leg
                gross_ret = (long_ret + short_ret) / 2.0
                # Trading costs: 2 sides for long entry/exit + 2 sides for short entry/exit
                gross_ret -= 2 * COST_PER_SIDE
            elif n_long > 0:
                gross_ret = long_ret
                gross_ret -= 2 * COST_PER_SIDE  # entry + exit
            else:
                gross_ret = short_ret
                gross_ret -= 2 * COST_PER_SIDE

            # Clip extreme returns
            gross_ret = np.clip(gross_ret, -0.5, 2.0)

            equity[name] *= (1 + gross_ret)
            curves[name].append({'date': pred_date, 'equity': equity[name], 'ret': gross_ret})
            period_entry[f'{name}_ret'] = gross_ret

        # BTC benchmark
        btc_ret = dg['btc_ret']
        equity['btc_bh'] *= (1 + btc_ret)
        curves['btc_bh'].append({'date': pred_date, 'equity': equity['btc_bh'], 'ret': btc_ret})
        period_entry['btc_ret'] = btc_ret

        period_log.append(period_entry)

    elapsed = time.time() - t_start

    # ═══ Compute Metrics ═══
    metrics = {}
    for name in list(STRATEGIES.keys()) + ['btc_bh']:
        df = pd.DataFrame(curves[name]).set_index('date')
        if df.empty:
            continue
        eq = df['equity']
        rets = df['ret']
        n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
        n_periods = len(rets)
        periods_per_year = 365.25 / HOLDING_DAYS

        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_yrs) - 1 if eq.iloc[-1] > 0 else -1
        vol = rets.std() * np.sqrt(periods_per_year)
        sharpe = (rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else 0
        sortino_d = rets[rets < 0].std()
        sortino = (rets.mean() / sortino_d * np.sqrt(periods_per_year)) if sortino_d > 0 else 0
        peak = eq.expanding().max()
        dd = (eq - peak) / peak
        max_dd = dd.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        hit_rate = (rets > 0).mean() * 100

        metrics[name] = {
            'label': STRATEGIES[name]['label'] if name in STRATEGIES else 'BTC B&H',
            'final_equity': eq.iloc[-1],
            'total_return': (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100,
            'cagr': cagr * 100,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_dd': max_dd * 100,
            'calmar': calmar,
            'vol': vol * 100,
            'hit_rate': hit_rate,
            'n_periods': n_periods,
        }

    # Print metrics table
    print(f"\n{'='*110}")
    print(f"  BACKTEST RESULTS — Pairwise Ranker v2 ({HOLDING_DAYS}d hold, {elapsed/60:.0f} min)")
    print(f"{'='*110}")
    print(f"{'Strategy':<20} {'Final$':>10} {'Total%':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD%':>7} {'Calmar':>7} {'Vol%':>6} {'Hit%':>6}")
    print(f"{'-'*110}")
    for name in list(STRATEGIES.keys()) + ['btc_bh']:
        if name not in metrics:
            continue
        m = metrics[name]
        print(f"{m['label']:<20} {m['final_equity']:>10,.0f} {m['total_return']:>7.1f}% "
              f"{m['cagr']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']:>6.1f}% {m['calmar']:>7.2f} {m['vol']:>5.1f}% {m['hit_rate']:>5.1f}%")
    print(f"{'='*110}")

    # ═══ Build Dashboard ═══
    print("\nBuilding dashboard...")
    build_dashboard(curves, metrics, period_log, elapsed)

    return curves, metrics


def build_dashboard(curves, metrics, period_log, elapsed):
    strat_names = list(STRATEGIES.keys()) + ['btc_bh']
    colors = {
        'long_q10': '#4CAF50', 'short_q10': '#f44336', 'ls_q10': '#2196F3',
        'long_q20': '#66BB6A', 'short_q20': '#EF5350', 'ls_q20': '#42A5F5',
        'btc_bh': '#FFD700',
    }
    labels = {name: (STRATEGIES[name]['label'] if name in STRATEGIES else 'BTC B&H')
              for name in strat_names}

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Equity Curves (log scale)',
            'Strategy Comparison (metrics)',
            'Drawdown',
            'Rank IC Over Time',
            'Rolling 6-Period Sharpe',
            'Per-Period Returns Distribution',
            'Cumulative Return by Strategy',
            'Annual Returns Heatmap',
        ],
        row_heights=[0.3, 0.25, 0.25, 0.2],
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "box"}],
            [{"type": "bar"}, {"type": "heatmap"}],
        ],
    )

    # 1. Equity curves
    for name in strat_names:
        df = pd.DataFrame(curves[name]).set_index('date')
        if df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df['equity'], name=labels[name],
            line=dict(color=colors[name], width=2.5 if 'ls' in name or name == 'btc_bh' else 1.5),
        ), row=1, col=1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=1)

    # 2. Metrics comparison bars
    metric_names = ['sharpe', 'cagr', 'calmar']
    for mi, mname in enumerate(metric_names):
        vals = [metrics[n][mname] for n in strat_names if n in metrics]
        names = [labels[n] for n in strat_names if n in metrics]
        clrs = [colors[n] for n in strat_names if n in metrics]
        fig.add_trace(go.Bar(
            x=names, y=vals, name=mname.upper(),
            marker_color=clrs, opacity=0.7 + mi * 0.1,
            text=[f'{v:.2f}' for v in vals], textposition='outside',
            showlegend=False,
        ), row=1, col=2)

    # 3. Drawdown
    for name in strat_names:
        df = pd.DataFrame(curves[name]).set_index('date')
        if df.empty:
            continue
        eq = df['equity']
        peak = eq.expanding().max()
        dd = (eq - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=f'{labels[name]} DD',
            line=dict(color=colors[name], width=1.5),
            fill='tozeroy' if 'ls_q10' in name else None,
            showlegend=False,
        ), row=2, col=1)

    # 4. Rank IC
    plog = pd.DataFrame(period_log).set_index('date')
    if 'rank_ic' in plog.columns:
        ic = plog['rank_ic']
        ic_roll = ic.rolling(6, min_periods=1).mean()
        fig.add_trace(go.Bar(
            x=ic.index, y=ic,
            marker_color=['#4CAF50' if v > 0 else '#f44336' for v in ic],
            opacity=0.3, name='Rank IC', showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=ic_roll.index, y=ic_roll,
            line=dict(color='#FFD700', width=2), name='IC 6-MA', showlegend=False,
        ), row=2, col=2)
        fig.add_hline(y=0, line_color='gray', row=2, col=2)

    # 5. Rolling Sharpe
    for name in ['ls_q10', 'ls_q20', 'short_q10', 'long_q10']:
        df = pd.DataFrame(curves[name]).set_index('date')
        if df.empty or len(df) < 6:
            continue
        rets = df['ret']
        roll_sharpe = rets.rolling(6).mean() / (rets.rolling(6).std() + 1e-10) * np.sqrt(365/HOLDING_DAYS)
        fig.add_trace(go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe, name=f'{labels[name]}',
            line=dict(color=colors[name], width=1.5), showlegend=False,
        ), row=3, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=3, col=1)

    # 6. Return distribution box
    for name in strat_names:
        df = pd.DataFrame(curves[name])
        if df.empty:
            continue
        fig.add_trace(go.Box(
            y=df['ret'] * 100, name=labels[name],
            marker_color=colors[name], boxmean='sd', showlegend=False,
        ), row=3, col=2)

    # 7. Cumulative return bars
    for name in strat_names:
        if name not in metrics:
            continue
        fig.add_trace(go.Bar(
            x=[labels[name]], y=[metrics[name]['total_return']],
            marker_color=colors[name], showlegend=False,
            text=[f"{metrics[name]['total_return']:.0f}%"], textposition='outside',
        ), row=4, col=1)

    # 8. Annual returns heatmap
    annual_data = {}
    for name in strat_names:
        df = pd.DataFrame(curves[name]).set_index('date')
        if df.empty:
            continue
        eq = df['equity']
        for yr in sorted(eq.index.year.unique()):
            y = eq[eq.index.year == yr]
            if len(y) > 1:
                annual_ret = (y.iloc[-1] / y.iloc[0] - 1) * 100
                if name not in annual_data:
                    annual_data[name] = {}
                annual_data[name][yr] = annual_ret

    if annual_data:
        years = sorted(set(yr for d in annual_data.values() for yr in d.keys()))
        z = []
        ylabels = []
        for name in strat_names:
            if name in annual_data:
                ylabels.append(labels[name])
                z.append([annual_data[name].get(yr, 0) for yr in years])
        fig.add_trace(go.Heatmap(
            z=z, x=[str(y) for y in years], y=ylabels,
            colorscale='RdYlGn', zmid=0,
            text=[[f'{v:.0f}%' for v in row] for row in z],
            texttemplate='%{text}', textfont=dict(size=10),
            colorbar=dict(title='Return%', len=0.2, y=0.12),
        ), row=4, col=2)

    # Summary text
    best = max(metrics.items(), key=lambda x: x[1]['sharpe'] if x[0] != 'btc_bh' else -999)
    summary = (
        f"Best Strategy: {best[1]['label']} (Sharpe {best[1]['sharpe']:.2f}, "
        f"CAGR {best[1]['cagr']:.1f}%, MaxDD {best[1]['max_dd']:.1f}%) | "
        f"BTC: Sharpe {metrics.get('btc_bh', {}).get('sharpe', 0):.2f} | "
        f"{len(period_log)} periods, {HOLDING_DAYS}d hold | {elapsed/60:.0f}min"
    )

    fig.update_layout(
        height=2400, width=1500, template='plotly_dark',
        title_text=f'Cross-Factor Pairwise Ranker v2 — Strategy Backtest<br><sub>{summary}</sub>',
        showlegend=True, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'backtest_strategies_v2.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"Dashboard: {path}")

    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')
    return path


if __name__ == '__main__':
    run_backtest()
