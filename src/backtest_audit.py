"""
Backtest Audit — Fix Leakage, Compare Two Approaches
=====================================================
A) Train label = fwd_ret_7d, purge=14d, evaluate with hold=28d
B) Train label = fwd_ret_28d, purge=35d, evaluate with hold=28d

Both use the same 6 strategy variants + BTC benchmark.
Proper multi-cohort handling: REBAL=28 (no overlap).
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

from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _train_model, _predict, _spearman_corr,
    _sample_pairs_and_train_epoch,
    N_EPOCHS, PAIRS_PER_DATE, NEAR_TIE_PCT, TAIL_WEIGHT_POW,
    LR, L1_REG, L2_REG, MIN_COINS, VOL_FLOOR_PCT,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

INITIAL_CAPITAL = 100_000
COST_PER_SIDE = 0.0007
FUNDING_DAILY = 0.0003

# Two configs to compare
CONFIGS = {
    'A_short_label': {
        'label': 'A: Train 7d label, purge 14d',
        'train_days': 540,
        'train_label_days': 7,    # SHORT label for training
        'purge_days': 14,         # purge >= train_label (7) + buffer (7)
        'hold_days': 28,          # evaluate on 28d
        'rebal_days': 28,         # no overlap
    },
    'B_long_label': {
        'label': 'B: Train 28d label, purge 35d',
        'train_days': 540,
        'train_label_days': 28,   # LONG label for training (same as hold)
        'purge_days': 35,         # purge >= train_label (28) + buffer (7)
        'hold_days': 28,          # evaluate on 28d
        'rebal_days': 28,         # no overlap
    },
}

STRATEGIES = {
    'long_q10':  {'long_pct': 0.10, 'short_pct': 0.00, 'label': 'Long 10%'},
    'short_q10': {'long_pct': 0.00, 'short_pct': 0.10, 'label': 'Short 10%'},
    'ls_q10':    {'long_pct': 0.10, 'short_pct': 0.10, 'label': 'L/S 10-10%'},
    'long_q20':  {'long_pct': 0.20, 'short_pct': 0.00, 'label': 'Long 20%'},
    'short_q20': {'long_pct': 0.00, 'short_pct': 0.20, 'label': 'Short 20%'},
    'ls_q20':    {'long_pct': 0.20, 'short_pct': 0.20, 'label': 'L/S 20-20%'},
}


def build_date_groups(panel, feat_cols, hold_days, train_label_days):
    """Build date groups with BOTH hold and train-label forward returns."""
    fwd_hold = f'fwd_ret_{hold_days}d'
    fwd_train = f'fwd_ret_{train_label_days}d'

    for col_name, days in [(fwd_hold, hold_days), (fwd_train, train_label_days)]:
        if col_name not in panel.columns:
            panel[col_name] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(days).shift(-days)
            )

    date_groups = {}
    for date, group in panel.groupby(level='date'):
        # Filter on close only — do NOT filter on fwd_ret (survivorship fix)
        g = group.dropna(subset=['close'])
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

        # Training labels (short or long horizon)
        train_ret = g[fwd_train].values.astype(np.float64) if fwd_train in g.columns else np.zeros(len(g))
        train_ret = np.nan_to_num(train_ret, nan=0.0)
        train_mkt = np.nanmean(train_ret) if np.any(np.isfinite(train_ret)) else 0.0
        train_excess = train_ret - train_mkt

        # Evaluation returns (always hold_days)
        eval_ret = g[fwd_hold].values.astype(np.float64) if fwd_hold in g.columns else np.zeros(len(g))
        eval_ret = np.nan_to_num(eval_ret, nan=0.0)

        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []
        btc_ret = 0.0
        if 'BTCUSDT' in syms:
            btc_ret = eval_ret[syms.index('BTCUSDT')]

        date_groups[date] = {
            'features': features,
            'train_excess': train_excess,   # for model training
            'train_ret': train_ret,         # raw train labels
            'eval_ret': eval_ret,           # for backtest evaluation
            'symbols': syms,
            'btc_ret': btc_ret,
            'n_coins': len(syms),
            'has_train_labels': np.sum(np.abs(train_ret) > 1e-10) >= MIN_COINS // 2,
            'has_eval_labels': np.sum(np.abs(eval_ret) > 1e-10) >= MIN_COINS // 2,
        }

    return date_groups


def run_single_config(config_name, cfg, panel, feat_cols):
    """Run backtest for a single config. Returns curves, metrics."""
    print(f"\n{'#'*60}")
    print(f"  {cfg['label']}")
    print(f"  train={cfg['train_days']}d, train_label={cfg['train_label_days']}d, "
          f"purge={cfg['purge_days']}d, hold={cfg['hold_days']}d, rebal={cfg['rebal_days']}d")
    print(f"{'#'*60}")

    date_groups = build_date_groups(panel, feat_cols, cfg['hold_days'], cfg['train_label_days'])
    sorted_dates = sorted(date_groups.keys())
    n_feat = len([f for f in feat_cols if f in panel.columns])
    print(f"  {len(sorted_dates)} valid dates, {n_feat} features")

    # Walk-forward schedule — no overlap (rebal = hold)
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if (d - sorted_dates[0]).days >= cfg['train_days'] + cfg['purge_days']:
            start_idx = i
            break

    rebal_dates = sorted_dates[start_idx::cfg['rebal_days']]
    print(f"  {len(rebal_dates)} rebalance dates ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # Backtest
    t0 = time.time()
    equity = {name: INITIAL_CAPITAL for name in STRATEGIES}
    equity['btc_bh'] = INITIAL_CAPITAL
    curves = {name: [] for name in list(STRATEGIES.keys()) + ['btc_bh']}
    period_log = []
    weights = np.zeros(n_feat)
    last_train_ri = None
    retrain_every = 1  # every rebalance

    for ri, pred_date in enumerate(rebal_dates):
        should_train = (last_train_ri is None or ri % retrain_every == 0)

        if should_train:
            train_end = pred_date - pd.Timedelta(days=cfg['purge_days'])
            train_start = train_end - pd.Timedelta(days=cfg['train_days'])

            train_dates = [d for d in sorted_dates
                          if train_start <= d <= train_end
                          and d in date_groups
                          and date_groups[d]['has_train_labels']]

            if len(train_dates) < 10:
                continue

            all_feat, all_eret, offsets = [], [], [0]
            for td in train_dates:
                dg = date_groups[td]
                # Only use rows with valid train labels
                mask = np.abs(dg['train_ret']) > 1e-10
                if np.sum(mask) < 5:
                    continue
                all_feat.append(dg['features'][mask])
                all_eret.append(dg['train_excess'][mask])
                offsets.append(offsets[-1] + int(np.sum(mask)))

            if len(all_feat) < 5:
                continue

            weights = _train_model(
                np.vstack(all_feat), np.concatenate(all_eret),
                np.array(offsets, dtype=np.int64),
                n_feat, N_EPOCHS, PAIRS_PER_DATE,
                NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG,
                42 + ri,
            )
            last_train_ri = ri

            if ri == 0 or (ri + 1) % 10 == 0:
                print(f"  [{ri+1}/{len(rebal_dates)}] {pred_date.date()} "
                      f"({len(train_dates)} train dates) [{time.time()-t0:.0f}s]", flush=True)

        if last_train_ri is None:
            continue
        if pred_date not in date_groups:
            continue

        dg = date_groups[pred_date]
        if not dg['has_eval_labels']:
            continue

        scores = _predict(dg['features'], weights)
        n_coins = dg['n_coins']
        eval_ret = dg['eval_ret']
        sorted_idx = np.argsort(scores)

        # Rank IC (scores vs excess of eval return — out of sample)
        eval_excess = eval_ret - np.mean(eval_ret)
        rank_ic = float(_spearman_corr(scores, eval_excess))

        entry = {'date': pred_date, 'rank_ic': rank_ic, 'n_coins': n_coins}

        for name, scfg in STRATEGIES.items():
            n_long = max(1, int(n_coins * scfg['long_pct'])) if scfg['long_pct'] > 0 else 0
            n_short = max(1, int(n_coins * scfg['short_pct'])) if scfg['short_pct'] > 0 else 0

            long_ret = np.mean(eval_ret[sorted_idx[-n_long:]]) if n_long > 0 else 0.0
            short_ret = 0.0
            if n_short > 0:
                short_ret = -np.mean(eval_ret[sorted_idx[:n_short]])
                short_ret -= FUNDING_DAILY * cfg['hold_days']

            if n_long > 0 and n_short > 0:
                gross_ret = (long_ret + short_ret) / 2.0 - 2 * COST_PER_SIDE
            elif n_long > 0:
                gross_ret = long_ret - 2 * COST_PER_SIDE
            else:
                gross_ret = short_ret - 2 * COST_PER_SIDE

            gross_ret = np.clip(gross_ret, -0.5, 2.0)
            equity[name] *= (1 + gross_ret)
            curves[name].append({'date': pred_date, 'equity': equity[name], 'ret': gross_ret})
            entry[f'{name}_ret'] = gross_ret

        # BTC
        equity['btc_bh'] *= (1 + dg['btc_ret'])
        curves['btc_bh'].append({'date': pred_date, 'equity': equity['btc_bh'], 'ret': dg['btc_ret']})
        entry['btc_ret'] = dg['btc_ret']
        period_log.append(entry)

    elapsed = time.time() - t0

    # Metrics
    metrics = {}
    for name in list(STRATEGIES.keys()) + ['btc_bh']:
        df = pd.DataFrame(curves[name]).set_index('date')
        if len(df) < 2:
            continue
        eq = df['equity']
        rets = df['ret']
        n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
        ppyr = 365.25 / cfg['hold_days']

        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_yrs) - 1 if eq.iloc[-1] > 0 else -1
        sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppyr)
        sortino_d = rets[rets < 0].std()
        sortino = rets.mean() / (sortino_d + 1e-10) * np.sqrt(ppyr)
        peak = eq.expanding().max()
        max_dd = ((eq - peak) / peak).min()
        calmar = cagr / (abs(max_dd) + 1e-10)
        hit = (rets > 0).mean() * 100

        metrics[name] = {
            'label': STRATEGIES[name]['label'] if name in STRATEGIES else 'BTC B&H',
            'final': eq.iloc[-1],
            'total_pct': (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100,
            'cagr': cagr * 100,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_dd': max_dd * 100,
            'calmar': calmar,
            'vol': rets.std() * np.sqrt(ppyr) * 100,
            'hit': hit,
            'n_periods': len(rets),
        }

    # IC stats
    plog = pd.DataFrame(period_log)
    avg_ic = plog['rank_ic'].mean() if len(plog) > 0 else 0
    ic_pos = (plog['rank_ic'] > 0).mean() * 100 if len(plog) > 0 else 0

    print(f"\n  {cfg['label']} — Results ({elapsed/60:.1f}min)")
    print(f"  Avg IC: {avg_ic:.4f} | IC>0: {ic_pos:.0f}% | Periods: {len(plog)}")
    print(f"  {'Strategy':<15} {'Final$':>10} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Hit%':>6}")
    print(f"  {'-'*60}")
    for name in list(STRATEGIES.keys()) + ['btc_bh']:
        if name not in metrics:
            continue
        m = metrics[name]
        print(f"  {m['label']:<15} {m['final']:>10,.0f} {m['cagr']:>6.1f}% "
              f"{m['sharpe']:>7.2f} {m['max_dd']:>6.1f}% {m['hit']:>5.1f}%")

    return curves, metrics, period_log, elapsed


def build_comparison_dashboard(all_results):
    """Build side-by-side comparison dashboard."""
    config_names = list(all_results.keys())
    n_configs = len(config_names)

    strat_names = list(STRATEGIES.keys()) + ['btc_bh']
    colors = {
        'long_q10': '#4CAF50', 'short_q10': '#f44336', 'ls_q10': '#2196F3',
        'long_q20': '#66BB6A', 'short_q20': '#EF5350', 'ls_q20': '#42A5F5',
        'btc_bh': '#FFD700',
    }
    labels = {n: (STRATEGIES[n]['label'] if n in STRATEGIES else 'BTC B&H') for n in strat_names}

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            f'{CONFIGS[config_names[0]]["label"]} — Equity',
            f'{CONFIGS[config_names[1]]["label"]} — Equity',
            f'{CONFIGS[config_names[0]]["label"]} — Drawdown',
            f'{CONFIGS[config_names[1]]["label"]} — Drawdown',
            'Sharpe Comparison',
            'CAGR Comparison',
            f'{CONFIGS[config_names[0]]["label"]} — IC',
            f'{CONFIGS[config_names[1]]["label"]} — IC',
        ],
        row_heights=[0.3, 0.25, 0.25, 0.2],
    )

    for ci, cname in enumerate(config_names):
        curves, metrics, plog, _ = all_results[cname]
        col = ci + 1

        # Equity
        for sname in strat_names:
            df = pd.DataFrame(curves[sname]).set_index('date')
            if df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=df.index, y=df['equity'], name=f'{labels[sname]}' if ci == 0 else None,
                line=dict(color=colors[sname], width=2 if 'ls' in sname else 1.2),
                showlegend=(ci == 0),
                legendgroup=sname,
            ), row=1, col=col)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=col)
        fig.update_yaxes(type='log', row=1, col=col)

        # Drawdown
        for sname in ['ls_q10', 'ls_q20', 'short_q10', 'long_q10']:
            df = pd.DataFrame(curves[sname]).set_index('date')
            if df.empty:
                continue
            eq = df['equity']
            dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd, name=None,
                line=dict(color=colors[sname], width=1.5),
                showlegend=False, legendgroup=sname,
            ), row=2, col=col)

        # IC
        pl = pd.DataFrame(plog)
        if 'rank_ic' in pl.columns and len(pl) > 0:
            pl = pl.set_index('date')
            ic = pl['rank_ic']
            ic_roll = ic.rolling(4, min_periods=1).mean()
            fig.add_trace(go.Bar(
                x=ic.index, y=ic,
                marker_color=['#4CAF50' if v > 0 else '#f44336' for v in ic],
                opacity=0.3, showlegend=False,
            ), row=4, col=col)
            fig.add_trace(go.Scatter(
                x=ic_roll.index, y=ic_roll,
                line=dict(color='#FFD700', width=2), showlegend=False,
            ), row=4, col=col)
            fig.add_hline(y=0, line_color='gray', row=4, col=col)
            avg_ic = ic.mean()
            fig.add_annotation(
                x=0.5, y=1.05,
                xref=f'x{4 + (ci*2 - 1) if ci > 0 else 3} domain' if ci == 0 else f'x{4 + ci*2} domain',
                yref=f'y{7 + ci} domain',
                text=f'IC={avg_ic:.4f} | IC>0={((ic>0).mean()*100):.0f}%',
                showarrow=False, font=dict(color='#FFD700', size=11),
            )

    # Sharpe comparison (row 3, col 1)
    x_labels = []
    sharpe_a = []
    sharpe_b = []
    for sname in strat_names:
        x_labels.append(labels[sname])
        sa = all_results[config_names[0]][1].get(sname, {}).get('sharpe', 0)
        sb = all_results[config_names[1]][1].get(sname, {}).get('sharpe', 0)
        sharpe_a.append(sa)
        sharpe_b.append(sb)

    fig.add_trace(go.Bar(
        x=x_labels, y=sharpe_a, name=CONFIGS[config_names[0]]['label'],
        marker_color='#03A9F4', opacity=0.7,
        text=[f'{v:.2f}' for v in sharpe_a], textposition='outside',
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=x_labels, y=sharpe_b, name=CONFIGS[config_names[1]]['label'],
        marker_color='#FF9800', opacity=0.7,
        text=[f'{v:.2f}' for v in sharpe_b], textposition='outside',
    ), row=3, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=3, col=1)

    # CAGR comparison (row 3, col 2)
    cagr_a = [all_results[config_names[0]][1].get(s, {}).get('cagr', 0) for s in strat_names]
    cagr_b = [all_results[config_names[1]][1].get(s, {}).get('cagr', 0) for s in strat_names]
    fig.add_trace(go.Bar(
        x=x_labels, y=cagr_a, name=None, showlegend=False,
        marker_color='#03A9F4', opacity=0.7,
        text=[f'{v:.1f}%' for v in cagr_a], textposition='outside',
    ), row=3, col=2)
    fig.add_trace(go.Bar(
        x=x_labels, y=cagr_b, name=None, showlegend=False,
        marker_color='#FF9800', opacity=0.7,
        text=[f'{v:.1f}%' for v in cagr_b], textposition='outside',
    ), row=3, col=2)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=3, col=2)

    # Summary
    best_a = max((m for n, m in all_results[config_names[0]][1].items() if n != 'btc_bh'),
                 key=lambda x: x['sharpe'], default={})
    best_b = max((m for n, m in all_results[config_names[1]][1].items() if n != 'btc_bh'),
                 key=lambda x: x['sharpe'], default={})

    summary = (
        f"A ({CONFIGS[config_names[0]]['label']}): Best={best_a.get('label','')} "
        f"Sharpe={best_a.get('sharpe',0):.2f} CAGR={best_a.get('cagr',0):.1f}% | "
        f"B ({CONFIGS[config_names[1]]['label']}): Best={best_b.get('label','')} "
        f"Sharpe={best_b.get('sharpe',0):.2f} CAGR={best_b.get('cagr',0):.1f}% | "
        f"PURGE FIX APPLIED — No label leakage"
    )

    fig.update_layout(
        height=2200, width=1500, template='plotly_dark',
        title_text=f'Audit Backtest: Short vs Long Training Labels (Leakage Fixed)<br><sub>{summary}</sub>',
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        barmode='group',
    )

    path = os.path.join(RESULTS_DIR, 'backtest_audit.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")
    return path


def main():
    os.chdir(PROJECT_DIR)
    panel = load_panel()
    feat_cols = get_feature_cols(panel)

    # JIT warmup
    n_feat = len([f for f in feat_cols if f in panel.columns])
    dummy = np.random.randn(20, n_feat)
    dummy_e = np.random.randn(20)
    dummy_w = np.zeros(n_feat)
    _sample_pairs_and_train_epoch(dummy, dummy_e, dummy_w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    all_results = {}
    for cname, cfg in CONFIGS.items():
        curves, metrics, plog, elapsed = run_single_config(cname, cfg, panel, feat_cols)
        all_results[cname] = (curves, metrics, plog, elapsed)

    # Build comparison dashboard
    path = build_comparison_dashboard(all_results)

    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')

    return all_results


if __name__ == '__main__':
    main()
