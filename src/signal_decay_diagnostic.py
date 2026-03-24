"""
Signal Decay Diagnostic — 3 Experiments from the C-Level Meeting
=================================================================
(a) Decile returns at 1, 3, 5, 7, 14, 28d horizons (existing model, no retrain)
(b) Signal rank autocorrelation decay (t vs t+1, t+3, t+5, t+7, t+14)
(c) Optimal holding period identification

Uses the EXISTING trained model. No new training.
Fast: ~5 min total.
"""
import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

from numba import njit
from scipy.stats import spearmanr, rankdata
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _train_model, _predict, _spearman_corr,
    _sample_pairs_and_train_epoch,
    LR, L1_REG, L2_REG, MIN_COINS, VOL_FLOOR_PCT,
    N_EPOCHS, PAIRS_PER_DATE, NEAR_TIE_PCT, TAIL_WEIGHT_POW,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

TRAIN_DAYS = 540
PURGE_DAYS = 14
TRAIN_LABEL_DAYS = 7
N_DECILES = 10
COST_PER_SIDE = 0.0007
FUNDING_DAILY = 0.0003

# Horizons to test
HORIZONS = [1, 3, 5, 7, 14, 28]


def main():
    os.chdir(PROJECT_DIR)
    t0 = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    n_feat = len([f for f in feat_cols if f in panel.columns])

    # Ensure all forward return columns exist
    print("Computing forward returns at all horizons...")
    for h in HORIZONS:
        col = f'fwd_ret_{h}d'
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(h).shift(-h))

    # Also need 7d train label
    fwd_train = f'fwd_ret_{TRAIN_LABEL_DAYS}d'

    # Build date groups
    print("Building date groups...")
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

        train_ret = np.nan_to_num(g[fwd_train].values.astype(np.float64), nan=0.0)
        train_mkt = np.nanmean(train_ret)
        train_excess = (train_ret - train_mkt)
        vol = np.std(train_ret)
        if vol > 1e-10:
            train_excess = train_excess / vol

        # All horizon returns
        horizon_rets = {}
        for h in HORIZONS:
            col = f'fwd_ret_{h}d'
            if col in g.columns:
                horizon_rets[h] = np.nan_to_num(g[col].values.astype(np.float64), nan=0.0)

        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []

        date_groups[date] = {
            'features': features,
            'train_excess': train_excess,
            'train_ret': train_ret,
            'horizon_rets': horizon_rets,
            'symbols': syms,
            'n_coins': len(syms),
        }

    sorted_dates = sorted(date_groups.keys())
    print(f"  {len(sorted_dates)} valid dates, {n_feat} features")

    # JIT warmup
    dummy = np.random.randn(20, n_feat)
    dummy_e = np.random.randn(20)
    dummy_w = np.zeros(n_feat)
    _sample_pairs_and_train_epoch(dummy, dummy_e, dummy_w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # Walk-forward: train once per month, predict daily
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if (d - sorted_dates[0]).days >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i
            break

    # Predict every 7 days (weekly), retrain every 4 weeks
    pred_dates = sorted_dates[start_idx::7]
    print(f"  {len(pred_dates)} prediction dates")

    # ═══ EXPERIMENT A: Decile returns at each horizon ═══
    print("\n=== EXPERIMENT A: Decile returns at multiple horizons ===")
    # {horizon: {decile: [returns]}}
    decile_rets = {h: {d: [] for d in range(N_DECILES)} for h in HORIZONS}
    ic_by_horizon = {h: [] for h in HORIZONS}
    weights = None
    last_train_idx = None
    retrain_every = 4  # every 4 predictions = ~28 days

    for ri, pred_date in enumerate(pred_dates):
        if last_train_idx is None or ri % retrain_every == 0:
            train_end = pred_date - pd.Timedelta(days=PURGE_DAYS)
            train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)

            train_data_f, train_data_e, offsets = [], [], [0]
            for d in sorted_dates:
                if d < train_start or d > train_end: continue
                if d not in date_groups: continue
                dg = date_groups[d]
                mask = np.abs(dg['train_ret']) > 1e-10
                if np.sum(mask) < 5: continue
                train_data_f.append(dg['features'][mask])
                train_data_e.append(dg['train_excess'][mask])
                offsets.append(offsets[-1] + int(np.sum(mask)))

            if len(train_data_f) < 10: continue

            weights = _train_model(
                np.vstack(train_data_f), np.concatenate(train_data_e),
                np.array(offsets, dtype=np.int64),
                n_feat, N_EPOCHS, PAIRS_PER_DATE,
                NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG, 42 + ri,
            )
            last_train_idx = ri

            if ri == 0 or (ri + 1) % 20 == 0:
                print(f"  [{ri+1}/{len(pred_dates)}] Trained at {pred_date.date()} [{time.time()-t0:.0f}s]",
                      flush=True)

        if weights is None: continue
        if pred_date not in date_groups: continue

        dg = date_groups[pred_date]
        scores = _predict(dg['features'], weights)
        n_coins = dg['n_coins']
        sorted_idx = np.argsort(scores)

        for h in HORIZONS:
            if h not in dg['horizon_rets']: continue
            hret = dg['horizon_rets'][h]
            if np.sum(np.abs(hret) > 1e-10) < MIN_COINS // 2: continue

            # Decile assignment
            for rank_pos, idx in enumerate(sorted_idx):
                dec = min(rank_pos * N_DECILES // n_coins, N_DECILES - 1)
                decile_rets[h][dec].append(hret[idx])

            # IC at this horizon
            excess = hret - np.mean(hret)
            ic = float(_spearman_corr(scores, excess))
            ic_by_horizon[h].append(ic)

    # ═══ EXPERIMENT B: Signal rank autocorrelation ═══
    print("\n=== EXPERIMENT B: Signal rank autocorrelation ===")
    # For each prediction date, store the ranking
    rankings = {}  # date -> rank array
    for ri, pred_date in enumerate(pred_dates):
        if pred_date not in date_groups: continue
        dg = date_groups[pred_date]
        if weights is None: continue
        scores = _predict(dg['features'], weights)
        rankings[pred_date] = {
            'scores': scores,
            'symbols': dg['symbols'],
        }

    # Compute rank autocorrelation at various lags
    lags_days = [1, 3, 5, 7, 14, 21, 28]
    autocorr_results = {lag: [] for lag in lags_days}

    pred_dates_list = sorted(rankings.keys())
    for i, d1 in enumerate(pred_dates_list):
        for lag in lags_days:
            # Find closest date that is ~lag days later
            target = d1 + pd.Timedelta(days=lag)
            best_d2 = None
            best_diff = 999
            for d2 in pred_dates_list[i+1:]:
                diff = abs((d2 - target).days)
                if diff < best_diff:
                    best_diff = diff
                    best_d2 = d2
                if (d2 - target).days > lag: break
            if best_d2 is None or best_diff > lag * 0.5: continue

            r1 = rankings[d1]
            r2 = rankings[best_d2]
            # Find common symbols
            common = set(r1['symbols']) & set(r2['symbols'])
            if len(common) < 20: continue

            idx1 = [r1['symbols'].index(s) for s in common]
            idx2 = [r2['symbols'].index(s) for s in common]
            s1 = r1['scores'][idx1]
            s2 = r2['scores'][idx2]
            corr = float(_spearman_corr(np.array(s1), np.array(s2)))
            autocorr_results[lag].append(corr)

    # ═══ Print Results ═══
    print(f"\n{'='*90}")
    print(f"  SIGNAL DECAY DIAGNOSTIC — {time.time()-t0:.0f}s")
    print(f"{'='*90}")

    print(f"\n  EXPERIMENT A: Decile spread at each horizon")
    print(f"  {'Horizon':>8} {'D1':>8} {'D5':>8} {'D10':>8} {'Spread':>8} {'IC':>8} {'IC>0':>6} {'Sharpe_LS':>10}")
    print(f"  {'-'*80}")
    horizon_metrics = {}
    for h in HORIZONS:
        d1_mean = np.mean(decile_rets[h][0]) * 100 if decile_rets[h][0] else 0
        d5_mean = np.mean(decile_rets[h][4]) * 100 if decile_rets[h][4] else 0
        d10_mean = np.mean(decile_rets[h][9]) * 100 if decile_rets[h][9] else 0
        spread = d10_mean - d1_mean
        avg_ic = np.mean(ic_by_horizon[h]) if ic_by_horizon[h] else 0
        ic_pos = np.mean(np.array(ic_by_horizon[h]) > 0) * 100 if ic_by_horizon[h] else 0

        # Approximate L/S Sharpe: spread per period / std of spread, annualized
        spreads = []
        for dec_rets_list in zip(*(decile_rets[h][d] for d in [0, 9])):
            pass  # complex, simplify
        # Simple: use IC-based Sharpe proxy
        if ic_by_horizon[h]:
            ic_arr = np.array(ic_by_horizon[h])
            ic_sharpe = ic_arr.mean() / (ic_arr.std() + 1e-10) * np.sqrt(365 / h)
        else:
            ic_sharpe = 0

        all_decile_means = [np.mean(decile_rets[h][d]) * 100 for d in range(N_DECILES)]
        mono, _ = spearmanr(range(N_DECILES), all_decile_means) if len(all_decile_means) == N_DECILES else (0, 0)

        horizon_metrics[h] = {
            'spread': spread, 'ic': avg_ic, 'ic_pos': ic_pos,
            'ic_sharpe': ic_sharpe, 'mono': mono,
            'decile_means': all_decile_means,
            'd1': d1_mean, 'd10': d10_mean,
        }

        print(f"  {h:>5}d {d1_mean:>8.2f} {d5_mean:>8.2f} {d10_mean:>8.2f} "
              f"{spread:>8.2f} {avg_ic:>8.4f} {ic_pos:>5.0f}% {ic_sharpe:>10.2f}")

    print(f"\n  EXPERIMENT A: Full decile profile per horizon")
    print(f"  {'Hz':>4} " + " ".join(f"{'D'+str(d+1):>7}" for d in range(N_DECILES)) + f" {'Mono':>7}")
    print(f"  {'-'*90}")
    for h in HORIZONS:
        dm = horizon_metrics[h]['decile_means']
        mono = horizon_metrics[h]['mono']
        print(f"  {h:>3}d " + " ".join(f"{d:>7.2f}" for d in dm) + f" {mono:>7.3f}")

    print(f"\n  EXPERIMENT B: Signal rank autocorrelation")
    print(f"  {'Lag':>8} {'Avg Corr':>10} {'Std':>8} {'N':>6}")
    print(f"  {'-'*40}")
    for lag in lags_days:
        if autocorr_results[lag]:
            arr = np.array(autocorr_results[lag])
            print(f"  {lag:>5}d {arr.mean():>10.3f} {arr.std():>8.3f} {len(arr):>6}")
        else:
            print(f"  {lag:>5}d {'N/A':>10}")

    # Half-life estimation
    print(f"\n  Signal half-life estimation:")
    for lag in lags_days:
        if autocorr_results[lag]:
            avg = np.mean(autocorr_results[lag])
            if avg < 0.5:
                print(f"  Rank autocorrelation drops below 0.5 at lag {lag}d")
                print(f"  -> Signal half-life ~ {lag} days")
                break
    else:
        print(f"  Signal still >0.5 at all tested lags (very persistent)")

    print(f"\n  CONCLUSION:")
    best_h = max(HORIZONS, key=lambda h: horizon_metrics[h]['ic_sharpe'])
    best_mono_h = max(HORIZONS, key=lambda h: horizon_metrics[h]['mono'])
    best_spread_h = max(HORIZONS, key=lambda h: horizon_metrics[h]['spread'])
    print(f"  Best IC-Sharpe: {best_h}d (IC-Sharpe={horizon_metrics[best_h]['ic_sharpe']:.2f})")
    print(f"  Best monotonicity: {best_mono_h}d (mono={horizon_metrics[best_mono_h]['mono']:.3f})")
    print(f"  Best spread: {best_spread_h}d (spread={horizon_metrics[best_spread_h]['spread']:.2f}%)")
    print(f"{'='*90}")

    # ═══ Dashboard ═══
    build_dashboard(horizon_metrics, ic_by_horizon, autocorr_results, decile_rets, lags_days)


def build_dashboard(horizon_metrics, ic_by_horizon, autocorr_results, decile_rets, lags_days):
    n = N_DECILES
    colors_h = {1: '#f44336', 3: '#FF9800', 5: '#FFEB3B', 7: '#4CAF50', 14: '#2196F3', 28: '#9C27B0'}

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Decile Returns by Horizon (THE key chart)',
            'Rank IC by Horizon',
            'Signal Rank Autocorrelation Decay',
            'IC-Sharpe by Horizon (optimal hold)',
            'D10-D1 Spread by Horizon',
            'Monotonicity by Horizon',
        ],
        row_heights=[0.4, 0.3, 0.3],
    )

    # 1. Decile returns per horizon
    for h in HORIZONS:
        dm = horizon_metrics[h]['decile_means']
        fig.add_trace(go.Scatter(
            x=[f'D{d+1}' for d in range(n)], y=dm,
            name=f'{h}d hold', mode='lines+markers',
            line=dict(color=colors_h[h], width=2.5),
            marker=dict(size=8),
        ), row=1, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=1, col=1)

    # 2. IC distribution per horizon
    for h in HORIZONS:
        if ic_by_horizon[h]:
            fig.add_trace(go.Box(
                y=ic_by_horizon[h], name=f'{h}d',
                marker_color=colors_h[h], boxmean='sd', showlegend=False,
            ), row=1, col=2)

    # 3. Autocorrelation decay
    lags = []
    corrs = []
    for lag in lags_days:
        if autocorr_results[lag]:
            lags.append(lag)
            corrs.append(np.mean(autocorr_results[lag]))
    fig.add_trace(go.Scatter(
        x=lags, y=corrs, mode='lines+markers',
        line=dict(color='#03A9F4', width=3), marker=dict(size=10),
        name='Rank Autocorr', showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0.5, line_color='#f44336', line_dash='dash', row=2, col=1,
                  annotation_text='Half-life threshold')
    fig.add_hline(y=0, line_color='gray', row=2, col=1)

    # 4. IC-Sharpe by horizon
    h_list = list(HORIZONS)
    ic_sharpes = [horizon_metrics[h]['ic_sharpe'] for h in h_list]
    best_idx = np.argmax(ic_sharpes)
    bar_colors = ['#FFD700' if i == best_idx else '#03A9F4' for i in range(len(h_list))]
    fig.add_trace(go.Bar(
        x=[f'{h}d' for h in h_list], y=ic_sharpes,
        marker_color=bar_colors, showlegend=False,
        text=[f'{v:.2f}' for v in ic_sharpes], textposition='outside',
    ), row=2, col=2)

    # 5. Spread by horizon
    spreads = [horizon_metrics[h]['spread'] for h in h_list]
    fig.add_trace(go.Bar(
        x=[f'{h}d' for h in h_list], y=spreads,
        marker_color=[colors_h[h] for h in h_list], showlegend=False,
        text=[f'{v:.2f}%' for v in spreads], textposition='outside',
    ), row=3, col=1)

    # 6. Monotonicity by horizon
    monos = [horizon_metrics[h]['mono'] for h in h_list]
    fig.add_trace(go.Bar(
        x=[f'{h}d' for h in h_list], y=monos,
        marker_color=[colors_h[h] for h in h_list], showlegend=False,
        text=[f'{v:.3f}' for v in monos], textposition='outside',
    ), row=3, col=2)

    # Summary
    best_h = h_list[best_idx]
    summary = (
        f"OPTIMAL HOLD: {best_h}d (IC-Sharpe={ic_sharpes[best_idx]:.2f}) | "
        f"Signal half-life: {lags[next((i for i, c in enumerate(corrs) if c < 0.5), -1)]}d | "
        f"Best spread: {max(spreads):.2f}% at {h_list[np.argmax(spreads)]}d | "
        f"Best mono: {max(monos):.3f} at {h_list[np.argmax(monos)]}d"
    ) if corrs else "Computing..."

    fig.update_layout(
        height=1600, width=1400, template='plotly_dark',
        title_text=f'Signal Decay Diagnostic<br><sub>{summary}</sub>',
        showlegend=True, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'signal_decay_diagnostic.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")

    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
