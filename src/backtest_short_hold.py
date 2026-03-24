"""
Short-Hold Backtest — 3d vs 5d vs 7d
======================================
Retrain models at each holding period.
Train label = hold period (aligned).
No overlap (rebal = hold).
Full 6-strategy backtest + BTC benchmark.
Ensemble: Linear + LGBMRanker Borda count.
"""
import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

from numba import njit
from scipy.stats import spearmanr, rankdata
from lightgbm import LGBMRanker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _train_model, _predict, _spearman_corr,
    _sample_pairs_and_train_epoch,
    LR, L1_REG, L2_REG, MIN_COINS, VOL_FLOOR_PCT,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

INITIAL_CAPITAL = 100_000
COST_PER_SIDE = 0.0007
FUNDING_DAILY = 0.0003
N_DECILES = 10
TRAIN_DAYS = 360       # shorter: recent data matters more for short holds
N_EPOCHS = 50
PAIRS_PER_DATE = 400
NEAR_TIE_PCT = 30.0
TAIL_WEIGHT_POW = 1.5
RETRAIN_EVERY = 4      # retrain every 4 rebalances

HOLD_CONFIGS = [
    {'hold': 3,  'purge': 7,  'label': '3d hold'},
    {'hold': 5,  'purge': 10, 'label': '5d hold'},
    {'hold': 7,  'purge': 14, 'label': '7d hold'},
]

STRATEGIES = {
    'long_q10':  {'lp': 0.10, 'sp': 0.00, 'label': 'Long 10%'},
    'short_q10': {'lp': 0.00, 'sp': 0.10, 'label': 'Short 10%'},
    'ls_q10':    {'lp': 0.10, 'sp': 0.10, 'label': 'L/S 10-10%'},
    'long_q20':  {'lp': 0.20, 'sp': 0.00, 'label': 'Long 20%'},
    'short_q20': {'lp': 0.00, 'sp': 0.20, 'label': 'Short 20%'},
    'ls_q20':    {'lp': 0.20, 'sp': 0.20, 'label': 'L/S 20-20%'},
}

LGBM_PARAMS = {
    'objective': 'lambdarank', 'metric': 'ndcg',
    'ndcg_eval_at': [3, 5], 'label_gain': [0, 1, 3, 7, 15],
    'n_estimators': 150, 'max_depth': 4, 'num_leaves': 12,
    'learning_rate': 0.03, 'min_child_samples': 15,
    'reg_alpha': 0.5, 'reg_lambda': 2.0,
    'subsample': 0.75, 'colsample_bytree': 0.6,
    'max_bin': 63, 'verbose': -1, 'random_state': 42, 'n_jobs': -1,
}


def run_single_hold(panel, feat_cols, hold_days, purge_days, config_label):
    """Run full backtest for a single holding period."""
    n_feat = len([f for f in feat_cols if f in panel.columns])
    fwd_col = f'fwd_ret_{hold_days}d'
    if fwd_col not in panel.columns:
        panel[fwd_col] = panel.groupby(level='symbol')['close'].transform(
            lambda x: x.pct_change(hold_days).shift(-hold_days))

    # Build date groups
    date_groups = {}
    for date, group in panel.groupby(level='date'):
        g = group.dropna(subset=['close'])
        if len(g) < MIN_COINS: continue
        if 'turnover_28d' in g.columns:
            thresh = g['turnover_28d'].quantile(VOL_FLOOR_PCT)
            g = g[g['turnover_28d'] >= thresh]
        if len(g) < MIN_COINS: continue

        available = [f for f in feat_cols if f in g.columns]
        features = g[available].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = _cross_sectional_rank_normalize(features)

        fwd = np.nan_to_num(g[fwd_col].values.astype(np.float64), nan=0.0)
        mkt = np.nanmean(fwd)
        vol = max(np.std(fwd), 1e-10)
        excess = (fwd - mkt) / vol  # vol-scaled excess

        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []
        btc_ret = fwd[syms.index('BTCUSDT')] if 'BTCUSDT' in syms else 0.0

        date_groups[date] = {
            'features': features, 'fwd_ret': fwd, 'excess': excess,
            'symbols': syms, 'n_coins': len(syms), 'btc_ret': btc_ret,
        }

    sorted_dates = sorted(date_groups.keys())

    # Walk-forward
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if (d - sorted_dates[0]).days >= TRAIN_DAYS + purge_days:
            start_idx = i; break

    rebal_dates = sorted_dates[start_idx::hold_days]  # no overlap
    print(f"  [{config_label}] {len(rebal_dates)} periods, {n_feat} features")

    # Results
    model_names = ['linear', 'lgbm', 'ensemble']
    equity = {mn: {sn: INITIAL_CAPITAL for sn in list(STRATEGIES.keys()) + ['btc_bh']}
              for mn in model_names}
    curves = {mn: {sn: [] for sn in list(STRATEGIES.keys()) + ['btc_bh']}
              for mn in model_names}
    period_log = {mn: [] for mn in model_names}

    w_linear = None
    m_lgbm = None
    t0 = time.time()

    for ri, pred_date in enumerate(rebal_dates):
        if ri % RETRAIN_EVERY == 0 or w_linear is None:
            train_end = pred_date - pd.Timedelta(days=purge_days)
            train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)

            tf, te, offsets = [], [], [0]
            tfl, tll, groups = [], [], []
            for d in sorted_dates:
                if d < train_start or d > train_end: continue
                if d not in date_groups: continue
                dg = date_groups[d]
                mask = np.abs(dg['fwd_ret']) > 1e-10
                if np.sum(mask) < 5: continue
                tf.append(dg['features'][mask])
                te.append(dg['excess'][mask])
                offsets.append(offsets[-1] + int(np.sum(mask)))
                # LGBM data
                tfl.append(dg['features'][mask])
                try:
                    labels = pd.qcut(dg['excess'][mask], 5, labels=False, duplicates='drop')
                except ValueError:
                    labels = np.clip((rankdata(dg['excess'][mask]) * 5 / (np.sum(mask) + 1)).astype(int), 0, 4)
                tll.append(labels)
                groups.append(int(np.sum(mask)))

            if len(tf) < 10: continue

            # Linear
            w_linear = _train_model(
                np.vstack(tf), np.concatenate(te),
                np.array(offsets, dtype=np.int64),
                n_feat, N_EPOCHS, PAIRS_PER_DATE,
                NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG, 42 + ri)

            # LGBM
            try:
                X_lgbm = np.vstack(tfl)
                y_lgbm = np.concatenate(tll).astype(int)
                sw = np.abs(np.concatenate(te)) + 0.1
                idx = 0
                for gs in groups:
                    s = sw[idx:idx+gs]
                    sw[idx:idx+gs] = s / (s.sum() + 1e-10) * gs
                    idx += gs
                m_lgbm = LGBMRanker(**LGBM_PARAMS)
                m_lgbm.fit(X_lgbm, y_lgbm, group=groups, sample_weight=sw)
            except Exception:
                m_lgbm = None

            if ri == 0 or (ri + 1) % 20 == 0:
                print(f"    [{ri+1}/{len(rebal_dates)}] {pred_date.date()} [{time.time()-t0:.0f}s]",
                      flush=True)

        if pred_date not in date_groups: continue
        dg = date_groups[pred_date]
        n_coins = dg['n_coins']
        fwd = dg['fwd_ret']

        # Winsorize individual coin returns (cap extreme outliers)
        fwd_w = np.clip(fwd, -0.5, 1.0)

        # Scores
        scores = {}
        if w_linear is not None:
            scores['linear'] = dg['features'] @ w_linear
        if m_lgbm is not None:
            scores['lgbm'] = m_lgbm.predict(dg['features'])
        if 'linear' in scores and 'lgbm' in scores:
            r1 = rankdata(scores['linear'])
            r2 = rankdata(scores['lgbm'])
            scores['ensemble'] = 0.4 * r1 + 0.6 * r2

        btc_ret = np.clip(dg['btc_ret'], -0.3, 0.5)

        for mn in model_names:
            if mn not in scores: continue
            sc = scores[mn]
            sorted_idx = np.argsort(sc)

            # IC
            excess = fwd_w - np.mean(fwd_w)
            rank_ic = float(_spearman_corr(sc, excess))

            # Decile means
            dm = np.zeros(N_DECILES)
            dc = np.zeros(N_DECILES)
            for rp, idx in enumerate(sorted_idx):
                db = min(rp * N_DECILES // n_coins, N_DECILES - 1)
                dm[db] += fwd_w[idx]
                dc[db] += 1
            dm = np.where(dc > 0, dm / dc, 0.0)

            period_log[mn].append({
                'date': pred_date, 'rank_ic': rank_ic,
                'decile_means': dm.tolist(), 'n_coins': n_coins,
            })

            for sn, scfg in STRATEGIES.items():
                nl = max(1, int(n_coins * scfg['lp'])) if scfg['lp'] > 0 else 0
                ns = max(1, int(n_coins * scfg['sp'])) if scfg['sp'] > 0 else 0

                lr = np.mean(fwd_w[sorted_idx[-nl:]]) if nl > 0 else 0.0
                sr = 0.0
                if ns > 0:
                    sr = -np.mean(fwd_w[sorted_idx[:ns]])
                    sr -= FUNDING_DAILY * hold_days

                if nl > 0 and ns > 0:
                    gr = (lr + sr) / 2.0 - 2 * COST_PER_SIDE
                elif nl > 0:
                    gr = lr - 2 * COST_PER_SIDE
                else:
                    gr = sr - 2 * COST_PER_SIDE

                gr = np.clip(gr, -0.3, 0.5)
                equity[mn][sn] *= (1 + gr)
                curves[mn][sn].append({'date': pred_date, 'equity': equity[mn][sn], 'ret': gr})

            equity[mn]['btc_bh'] *= (1 + btc_ret)
            curves[mn]['btc_bh'].append({'date': pred_date, 'equity': equity[mn]['btc_bh'], 'ret': btc_ret})

    return curves, period_log, time.time() - t0


def compute_metrics(curves, hold_days):
    metrics = {}
    ppyr = 365.25 / hold_days
    for mn in curves:
        metrics[mn] = {}
        for sn in curves[mn]:
            df = pd.DataFrame(curves[mn][sn]).set_index('date')
            if len(df) < 3: continue
            eq, rets = df['equity'], df['ret']
            n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
            cagr = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1/n_yrs) - 1) * 100
            sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppyr)
            mdd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100
            hit = (rets > 0).mean() * 100
            label = STRATEGIES[sn]['label'] if sn in STRATEGIES else 'BTC B&H'
            metrics[mn][sn] = {
                'label': label, 'final': eq.iloc[-1], 'cagr': cagr,
                'sharpe': sharpe, 'mdd': mdd, 'hit': hit, 'n': len(rets),
            }
    return metrics


def main():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)

    # JIT warmup
    n_feat = len([f for f in feat_cols if f in panel.columns])
    d, e, w = np.random.randn(20, n_feat), np.random.randn(20), np.zeros(n_feat)
    _sample_pairs_and_train_epoch(d, e, w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # Ensure all fwd columns
    for cfg in HOLD_CONFIGS:
        h = cfg['hold']
        col = f'fwd_ret_{h}d'
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(h).shift(-h))

    all_results = {}
    for cfg in HOLD_CONFIGS:
        print(f"\n{'#'*60}")
        print(f"  {cfg['label']} (hold={cfg['hold']}d, purge={cfg['purge']}d)")
        print(f"{'#'*60}")
        curves, plog, elapsed = run_single_hold(
            panel, feat_cols, cfg['hold'], cfg['purge'], cfg['label'])
        metrics = compute_metrics(curves, cfg['hold'])
        all_results[cfg['label']] = {
            'curves': curves, 'plog': plog, 'metrics': metrics,
            'hold': cfg['hold'], 'elapsed': elapsed,
        }

        # Print
        for mn in ['ensemble', 'linear', 'lgbm']:
            if mn not in metrics: continue
            plog_df = pd.DataFrame(plog.get(mn, []))
            avg_ic = plog_df['rank_ic'].mean() if len(plog_df) > 0 else 0
            ic_pos = (plog_df['rank_ic'] > 0).mean() * 100 if len(plog_df) > 0 else 0
            agg_dm = np.mean(plog_df['decile_means'].tolist(), axis=0) if len(plog_df) > 0 else np.zeros(10)
            mono, _ = spearmanr(range(N_DECILES), agg_dm) if len(agg_dm) == N_DECILES else (0, 0)

            print(f"\n  [{mn.upper()}] IC={avg_ic:.4f} IC>0={ic_pos:.0f}% Mono={mono:.3f}")
            print(f"  Deciles: {' '.join(f'{d*100:6.2f}' for d in agg_dm)}")
            print(f"  {'Strat':<15} {'$':>9} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} {'Hit%':>6}")
            for sn in list(STRATEGIES.keys()) + ['btc_bh']:
                if sn not in metrics[mn]: continue
                m = metrics[mn][sn]
                print(f"  {m['label']:<15} {m['final']:>9,.0f} {m['cagr']:>6.1f}% "
                      f"{m['sharpe']:>7.2f} {m['mdd']:>6.1f}% {m['hit']:>5.1f}%")

    total = time.time() - t_start
    print(f"\nTotal: {total/60:.1f} min")

    # Dashboard
    build_comparison_dashboard(all_results)


def build_comparison_dashboard(all_results):
    configs = list(all_results.keys())
    n_cfg = len(configs)
    colors_h = {'3d hold': '#FF9800', '5d hold': '#4CAF50', '7d hold': '#2196F3'}
    colors_m = {'linear': '#03A9F4', 'lgbm': '#FF9800', 'ensemble': '#9C27B0'}
    strat_list = list(STRATEGIES.keys()) + ['btc_bh']

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[
            '3d Hold — Equity (Ensemble)', '5d Hold — Equity (Ensemble)', '7d Hold — Equity (Ensemble)',
            '3d Hold — Deciles', '5d Hold — Deciles', '7d Hold — Deciles',
            'Sharpe Comparison (Ensemble)', 'IC Comparison', 'Monotonicity Comparison',
            'L/S 10-10% Across Holds', 'Short 10% Across Holds', 'Drawdown L/S 10-10%',
        ],
        row_heights=[0.3, 0.25, 0.2, 0.25],
    )

    colors_strat = {
        'long_q10': '#4CAF50', 'short_q10': '#f44336', 'ls_q10': '#2196F3',
        'long_q20': '#66BB6A', 'short_q20': '#EF5350', 'ls_q20': '#42A5F5',
        'btc_bh': '#FFD700',
    }

    for ci, cfg_name in enumerate(configs):
        res = all_results[cfg_name]
        col = ci + 1
        mn = 'ensemble'

        # Row 1: Equity
        for sn in ['ls_q10', 'short_q10', 'long_q10', 'btc_bh']:
            df = pd.DataFrame(res['curves'].get(mn, {}).get(sn, [])).set_index('date')
            if df.empty: continue
            label = STRATEGIES[sn]['label'] if sn in STRATEGIES else 'BTC'
            fig.add_trace(go.Scatter(
                x=df.index, y=df['equity'], name=label if ci == 0 else None,
                line=dict(color=colors_strat[sn], width=2),
                showlegend=(ci == 0), legendgroup=sn,
            ), row=1, col=col)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=col)
        fig.update_yaxes(type='log', row=1, col=col)

        # Row 2: Decile bars
        plog_df = pd.DataFrame(res['plog'].get(mn, []))
        if len(plog_df) > 0:
            agg_dm = np.mean(plog_df['decile_means'].tolist(), axis=0) * 100
            dcolors = [f'rgb({int(255*(1-i/9))}, {int(200*i/9)}, 80)' for i in range(10)]
            fig.add_trace(go.Bar(
                x=[f'D{d+1}' for d in range(10)], y=agg_dm,
                marker_color=dcolors, showlegend=False,
                text=[f'{v:.2f}' for v in agg_dm], textposition='outside',
            ), row=2, col=col)
            fig.add_hline(y=0, line_dash='dot', line_color='gray', row=2, col=col)

    # Row 3: Cross-hold comparisons
    # Sharpe
    for cfg_name in configs:
        res = all_results[cfg_name]
        mn = 'ensemble'
        sharpes = []
        labels = []
        for sn in strat_list:
            m = res['metrics'].get(mn, {}).get(sn, {})
            sharpes.append(m.get('sharpe', 0))
            labels.append(STRATEGIES[sn]['label'] if sn in STRATEGIES else 'BTC')
        fig.add_trace(go.Bar(
            x=labels, y=sharpes, name=cfg_name,
            marker_color=colors_h.get(cfg_name, '#888'), opacity=0.8,
        ), row=3, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='gray', row=3, col=1)

    # IC
    for cfg_name in configs:
        res = all_results[cfg_name]
        for mn in ['ensemble']:
            plog_df = pd.DataFrame(res['plog'].get(mn, []))
            if plog_df.empty: continue
            ic_roll = plog_df['rank_ic'].rolling(6, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=plog_df['date'], y=ic_roll, name=cfg_name,
                line=dict(color=colors_h.get(cfg_name, '#888'), width=2),
                showlegend=False,
            ), row=3, col=2)
    fig.add_hline(y=0, line_color='gray', row=3, col=2)

    # Monotonicity
    for cfg_name in configs:
        res = all_results[cfg_name]
        mn = 'ensemble'
        plog_df = pd.DataFrame(res['plog'].get(mn, []))
        if plog_df.empty: continue
        agg = np.mean(plog_df['decile_means'].tolist(), axis=0)
        mono, _ = spearmanr(range(10), agg)
        fig.add_trace(go.Bar(
            x=[cfg_name], y=[mono],
            marker_color=colors_h.get(cfg_name, '#888'), showlegend=False,
            text=[f'{mono:.3f}'], textposition='outside',
        ), row=3, col=3)

    # Row 4: Cross-hold equity for key strategies
    for cfg_name in configs:
        res = all_results[cfg_name]
        mn = 'ensemble'
        for sn, col_idx in [('ls_q10', 1), ('short_q10', 2)]:
            df = pd.DataFrame(res['curves'].get(mn, {}).get(sn, [])).set_index('date')
            if df.empty: continue
            fig.add_trace(go.Scatter(
                x=df.index, y=df['equity'], name=cfg_name if col_idx == 1 else None,
                line=dict(color=colors_h.get(cfg_name, '#888'), width=2.5),
                showlegend=(col_idx == 1), legendgroup=f'h_{cfg_name}',
            ), row=4, col=col_idx)

        # Drawdown
        df = pd.DataFrame(res['curves'].get(mn, {}).get('ls_q10', [])).set_index('date')
        if not df.empty:
            eq = df['equity']
            dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd, name=None,
                line=dict(color=colors_h.get(cfg_name, '#888'), width=1.5),
                showlegend=False,
            ), row=4, col=3)

    fig.update_layout(
        height=2200, width=1600, template='plotly_dark', barmode='group',
        title_text='Short-Hold Backtest: 3d vs 5d vs 7d (Ensemble: Linear + LGBMRanker)',
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'backtest_short_hold.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
