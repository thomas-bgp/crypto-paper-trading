"""
Decile Calibration v2 — Pairwise Ranking with Numba JIT
=========================================================
Implements the VertoxQuant LTR framework:
  - Pairwise logistic loss (RankNet-style) with JIT compilation
  - Tail-weighted pairs (50% tail-biased, 50% uniform)
  - Near-tie filtering (bottom 30th percentile of |Δy| ignored)
  - Excess return (market-neutralized) labels
  - Cross-sectional rank normalization of features
  - Walk-forward decile calibration

Near-Rust speed via Numba JIT on all hot loops.
"""
import numpy as np
import pandas as pd
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from numba import njit, prange
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Config (optimized from 432-config grid search) ───
TRAIN_DAYS = 540
PURGE_DAYS = 7
HOLDING_DAYS = 28
REBAL_EVERY = 7
MIN_COINS = 20
N_DECILES = 10
VOL_FLOOR_PCT = 0.20

# Pairwise model
LR = 0.002
L1_REG = 0.0             # Grid search result: L1=0 is optimal
L2_REG = 0.01            # Grid search result: L2=0.01 sweet spot
N_EPOCHS = 80
PAIRS_PER_DATE = 600
NEAR_TIE_PCT = 30.0      # ignore bottom 30th pct of |Δy|
TAIL_WEIGHT_POW = 1.0    # w ∝ |Δy|^pow
RETRAIN_EVERY = 1         # retrain every rebalance (grid search optimal)


# ════════════════════════════════════════════
# NUMBA JIT KERNELS — hot inner loops
# ════════════════════════════════════════════

@njit(cache=True)
def _sigmoid(x):
    if x > 20.0:
        return 1.0
    elif x < -20.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def _rankdata(x):
    """Simple rank (no tie-handling needed for continuous features)."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n)
    for i in range(n):
        ranks[order[i]] = i + 1.0
    return ranks


@njit(cache=True)
def _cross_sectional_rank_normalize(features):
    """Rank-normalize each feature column within the cross-section to [0,1]."""
    n, p = features.shape
    out = np.empty_like(features)
    for j in range(p):
        ranks = _rankdata(features[:, j])
        for i in range(n):
            out[i, j] = ranks[i] / n
    return out


@njit(cache=True)
def _percentile(arr, pct):
    """Simple percentile on sorted-ish data."""
    s = np.sort(arr)
    idx = int(pct / 100.0 * (len(s) - 1) + 0.5)
    if idx >= len(s):
        idx = len(s) - 1
    return s[idx]


@njit(cache=True)
def _sample_pairs_and_train_epoch(
    features,       # (n_coins, n_feat)
    excess_ret,     # (n_coins,)
    weights,        # (n_feat,) — model weights, MODIFIED in-place
    n_pairs,
    near_tie_pct,
    tail_pow,
    lr,
    l1_reg,
    l2_reg,
    rng_seed,
):
    """Sample tail-weighted pairs, compute pairwise logistic gradient, update weights.
    Returns (total_loss, n_valid_pairs).
    """
    n, p = features.shape
    if n < 4:
        return 0.0, 0

    # Pre-compute scores
    scores = features @ weights

    # Sort indices by excess return for tail sampling
    sorted_idx = np.argsort(excess_ret)
    tail_size = max(2, int(n * 0.2))
    top_start = n - tail_size

    # Compute epsilon for near-tie filtering via LCG RNG
    seed = rng_seed
    n_diff_samples = min(500, n * (n - 1) // 2)
    diffs = np.empty(n_diff_samples)
    for k in range(n_diff_samples):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        i = int(seed % n)
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        j = int(seed % n)
        if i == j:
            j = (j + 1) % n
        diffs[k] = abs(excess_ret[i] - excess_ret[j])
    epsilon = _percentile(diffs, near_tie_pct)

    # Sample and accumulate gradient
    grad = np.zeros(p)
    total_loss = 0.0
    valid_pairs = 0

    for k in range(n_pairs):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

        # 50% tail-biased, 50% uniform
        if (seed >> 32) % 2 == 0:
            # Tail: one from top, one from bottom
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            i = sorted_idx[top_start + int(seed % tail_size)]
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            j = sorted_idx[int(seed % tail_size)]
        else:
            # Uniform
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            i = int(seed % n)
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            j = int(seed % n)

        if i == j:
            continue

        abs_diff = abs(excess_ret[i] - excess_ret[j])
        if abs_diff < epsilon:
            continue  # near-tie filter

        # Tail weight
        w = abs_diff ** tail_pow

        # Preference
        sigma = 1.0 if excess_ret[i] > excess_ret[j] else -1.0
        diff = scores[i] - scores[j]
        margin = sigma * diff

        # Loss
        if margin > 20.0:
            loss = 0.0
        elif margin < -20.0:
            loss = -margin
        else:
            loss = np.log(1.0 + np.exp(-margin))
        total_loss += w * loss

        # Gradient: -w * sigma * sigmoid(-margin) * (x_i - x_j)
        sig = _sigmoid(-margin)
        coeff = -w * sigma * sig
        for f in range(p):
            grad[f] += coeff * (features[i, f] - features[j, f])
        valid_pairs += 1

    if valid_pairs > 0:
        scale = 1.0 / valid_pairs
        for f in range(p):
            # Elastic Net: L2 gradient + L1 subgradient
            l1_grad = l1_reg * (1.0 if weights[f] > 0 else (-1.0 if weights[f] < 0 else 0.0))
            g = grad[f] * scale + 2.0 * l2_reg * weights[f] + l1_grad
            weights[f] -= lr * g

    return total_loss, valid_pairs


@njit(cache=True)
def _train_model(
    all_features,     # list-like: concatenated features per date
    all_excess_ret,   # list-like: concatenated excess returns per date
    date_offsets,     # (n_dates+1,): start index of each date in the concatenated arrays
    n_feat,
    n_epochs,
    n_pairs,
    near_tie_pct,
    tail_pow,
    lr,
    l1_reg,
    l2_reg,
    seed,
):
    """Train pairwise ranker on multiple cross-sections. Returns weights."""
    weights = np.zeros(n_feat)
    n_dates = len(date_offsets) - 1

    for epoch in range(n_epochs):
        for d in range(n_dates):
            start = date_offsets[d]
            end = date_offsets[d + 1]
            n = end - start
            if n < 5:
                continue
            feat = all_features[start:end]
            eret = all_excess_ret[start:end]
            _sample_pairs_and_train_epoch(
                feat, eret, weights,
                n_pairs, near_tie_pct, tail_pow, lr, l1_reg, l2_reg,
                seed + epoch * 1000 + d,
            )

    return weights


@njit(cache=True)
def _predict(features, weights):
    return features @ weights


@njit(cache=True)
def _spearman_corr(x, y):
    """Fast Spearman correlation."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    n = len(x)
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += rx[i]
        my += ry[i]
    mx /= n
    my /= n
    cov = 0.0
    vx = 0.0
    vy = 0.0
    for i in range(n):
        dx = rx[i] - mx
        dy = ry[i] - my
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy
    denom = (vx * vy) ** 0.5
    if denom < 1e-15:
        return 0.0
    return cov / denom


# ════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════

def load_panel():
    """Load feature panel v2. Generate if missing."""
    panel_path = os.path.join(DATA_DIR, 'feature_panel_v2.parquet')
    if not os.path.exists(panel_path):
        print("Generating feature_panel_v2.parquet (200+ features)...")
        from feature_engine_v2 import build_feature_matrix_v2
        panel, _ = build_feature_matrix_v2(resample='1D', min_candles=250)
        panel.to_parquet(panel_path)
        return panel

    print("Loading feature_panel_v2.parquet...")
    panel = pd.read_parquet(panel_path)
    print(f"  Panel: {panel.shape}")
    return panel


def get_feature_cols(panel):
    exclude_prefixes = ('fwd_ret_', 'fwd_')
    exclude_exact = {
        'open', 'high', 'low', 'close', 'volume', 'quote_vol',
        'log_ret', 'spread_proxy', 'atr_14d', 'symbol',
    }
    cols = []
    for c in panel.columns:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if panel[c].dtype in ('float64', 'float32', 'int64'):
            cols.append(c)
    return cols


# ════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ════════════════════════════════════════════

def run_walk_forward(panel, feat_cols):
    """Walk-forward pairwise ranking with decile analysis."""
    fwd_col = f'fwd_ret_{HOLDING_DAYS}d'
    if fwd_col not in panel.columns:
        panel[fwd_col] = panel.groupby(level='symbol')['close'].transform(
            lambda x: x.pct_change(HOLDING_DAYS).shift(-HOLDING_DAYS)
        )

    # Get all dates
    all_dates = panel.index.get_level_values('date').unique().sort_values()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Group panel by date for fast access
    print("Grouping by date...")
    date_groups = {}
    for date, group in panel.groupby(level='date'):
        g = group.dropna(subset=[fwd_col, 'close'])
        if len(g) < MIN_COINS:
            continue

        # Volume filter
        if 'turnover_28d' in g.columns:
            thresh = g['turnover_28d'].quantile(VOL_FLOOR_PCT)
            g = g[g['turnover_28d'] >= thresh]

        if len(g) < MIN_COINS:
            continue

        # Extract numpy arrays
        available = [f for f in feat_cols if f in g.columns]
        features = g[available].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Cross-sectional rank normalization
        features = _cross_sectional_rank_normalize(features)

        fwd_ret = g[fwd_col].values.astype(np.float64)
        market_ret = np.nanmean(fwd_ret)
        excess_ret = fwd_ret - market_ret

        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else list(range(len(g)))

        date_groups[date] = {
            'features': features,
            'fwd_ret': fwd_ret,
            'excess_ret': excess_ret,
            'symbols': syms,
        }

    sorted_dates = sorted(date_groups.keys())
    n_dates = len(sorted_dates)
    n_feat = len([f for f in feat_cols if f in panel.columns])
    print(f"  {n_dates} valid dates, {n_feat} features")

    # Walk-forward
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        days_from_start = (d - sorted_dates[0]).days
        if days_from_start >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i
            break

    rebal_dates = sorted_dates[start_idx::REBAL_EVERY]
    print(f"  {len(rebal_dates)} prediction dates ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # JIT warmup
    print("JIT warmup...")
    dummy_feat = np.random.randn(20, n_feat)
    dummy_eret = np.random.randn(20)
    dummy_w = np.zeros(n_feat)
    _sample_pairs_and_train_epoch(dummy_feat, dummy_eret, dummy_w, 10, 30.0, 1.0, 0.001, 0.005, 0.01, 42)
    _spearman_corr(dummy_eret, dummy_eret)
    _cross_sectional_rank_normalize(dummy_feat)

    print("Running walk-forward...")
    t0 = time.time()

    all_records = []
    period_stats = []
    weights = np.zeros(n_feat)
    last_train_ri = None

    for ri, pred_date in enumerate(rebal_dates):
        should_train = (last_train_ri is None or ri % RETRAIN_EVERY == 0)

        if should_train:
            # Collect training dates
            train_end = pred_date - pd.Timedelta(days=PURGE_DAYS)
            train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)

            train_dates = [d for d in sorted_dates
                          if train_start <= d <= train_end and d in date_groups]

            if len(train_dates) < 10:
                continue

            # Concatenate training data for Numba
            all_feat_list = []
            all_eret_list = []
            offsets = [0]
            for td in train_dates:
                dg = date_groups[td]
                all_feat_list.append(dg['features'])
                all_eret_list.append(dg['excess_ret'])
                offsets.append(offsets[-1] + len(dg['excess_ret']))

            all_features = np.vstack(all_feat_list)
            all_excess_ret = np.concatenate(all_eret_list)
            date_offsets = np.array(offsets, dtype=np.int64)

            # Train
            weights = _train_model(
                all_features, all_excess_ret, date_offsets,
                n_feat, N_EPOCHS, PAIRS_PER_DATE,
                NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG,
                42 + ri,
            )
            last_train_ri = ri

            if ri == 0 or (ri + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{ri+1}/{len(rebal_dates)}] Trained on {len(train_dates)} dates "
                      f"({train_start.date()} to {train_end.date()}) [{elapsed:.1f}s]")

        if last_train_ri is None:
            continue

        # Predict
        if pred_date not in date_groups:
            continue
        dg = date_groups[pred_date]
        scores = _predict(dg['features'], weights)
        n_coins = len(scores)

        # Rank IC (scores vs excess return)
        rank_ic = float(_spearman_corr(scores, dg['excess_ret']))

        # Bin into deciles
        sorted_idx = np.argsort(scores)
        decile_sums = np.zeros(N_DECILES)
        decile_counts = np.zeros(N_DECILES, dtype=np.int64)
        for rank_pos, idx in enumerate(sorted_idx):
            decile = min(rank_pos * N_DECILES // n_coins, N_DECILES - 1)
            decile_sums[decile] += dg['fwd_ret'][idx]  # raw return for economic eval
            decile_counts[decile] += 1

        decile_means = np.where(decile_counts > 0, decile_sums / decile_counts, 0.0)
        date_str = str(pred_date.date())

        for d in range(N_DECILES):
            all_records.append({
                'date_str': date_str,
                'decile': d + 1,
                'mean_fwd_ret': decile_means[d],
                'n_coins': int(decile_counts[d]),
            })

        top_ret = decile_means[N_DECILES - 1]
        bot_ret = decile_means[0]
        period_stats.append({
            'date_str': date_str,
            'rank_ic': rank_ic,
            'n_coins': n_coins,
            'top_decile_ret': top_ret,
            'bot_decile_ret': bot_ret,
            'spread': top_ret - bot_ret,
        })

    elapsed = time.time() - t0

    # Feature importance
    feat_names = [f for f in feat_cols if f in panel.columns]
    importance = sorted(
        [{'name': feat_names[i], 'weight_abs': abs(weights[i]), 'weight_signed': weights[i]}
         for i in range(len(feat_names))],
        key=lambda x: -x['weight_abs']
    )
    n_active = sum(1 for w in weights if abs(w) > 1e-6)
    n_total = len(weights)
    print(f"  Elastic Net sparsity: {n_active}/{n_total} features active ({n_active/n_total*100:.0f}%)")

    records_df = pd.DataFrame(all_records)
    stats_df = pd.DataFrame(period_stats)

    if stats_df.empty:
        print("ERROR: No results!")
        return records_df, stats_df, importance, elapsed

    # Print summary
    avg_ic = stats_df['rank_ic'].mean()
    ic_pos = (stats_df['rank_ic'] > 0).mean() * 100
    avg_top = stats_df['top_decile_ret'].mean()
    avg_bot = stats_df['bot_decile_ret'].mean()
    avg_spread = stats_df['spread'].mean()

    # Aggregate decile means
    agg = records_df.groupby('decile')['mean_fwd_ret'].mean()
    mono, _ = spearmanr(range(len(agg)), agg.values)

    print(f"\n{'='*70}")
    print(f"  PAIRWISE RANKING v2 — DECILE CALIBRATION (Numba JIT)")
    print(f"{'='*70}")
    print(f"  Periods: {len(stats_df)}")
    print(f"  Avg Rank IC: {avg_ic:.4f}")
    print(f"  IC > 0: {ic_pos:.1f}%")
    print(f"  Avg Top Decile: {avg_top*100:.2f}%")
    print(f"  Avg Bot Decile: {avg_bot*100:.2f}%")
    print(f"  Avg Spread: {avg_spread*100:.2f}%")
    print(f"  Monotonicity: {mono:.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    print(f"\n  DECILE TABLE ({HOLDING_DAYS}d forward return):")
    print(f"  {'Decile':>8} {'Mean%':>8} {'Hit%':>8}")
    print(f"  {'-'*30}")
    for d in range(1, N_DECILES + 1):
        sub = records_df[records_df['decile'] == d]['mean_fwd_ret']
        print(f"  D{d:<7} {sub.mean()*100:>8.2f} {(sub > 0).mean()*100:>8.1f}")
    print(f"  {'-'*30}")
    print(f"  D10-D1 Spread: {(agg.iloc[-1] - agg.iloc[0])*100:.2f}%")

    print(f"\n  TOP 20 FEATURES:")
    for fi in importance[:20]:
        print(f"    {fi['name']:30s} {fi['weight_abs']:.4f} (signed: {fi['weight_signed']:+.4f})")
    print(f"{'='*70}")

    return records_df, stats_df, importance, elapsed


# ════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════

def build_dashboard(records_df, stats_df, importance, elapsed):
    n = N_DECILES
    colors = [f'rgb({int(255*(1-i/(n-1)))}, {int(200*i/(n-1))}, 80)' for i in range(n)]

    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            'Avg Forward Return by Decile',
            'Feature Importance (Signed Weights, Top 25)',
            'Rank IC Over Time',
            'IC Distribution',
            'Decile Return Heatmap',
            'Cumulative L/S Spread',
            'Top vs Bottom Decile Over Time',
            'Return Distribution per Decile',
            'Hit Rate by Decile',
            'Decile Sharpe Approximation',
        ],
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "box"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
    )

    # 1. Decile bar
    agg = records_df.groupby('decile')['mean_fwd_ret'].mean()
    mono, _ = spearmanr(range(len(agg)), agg.values)
    spread = (agg.iloc[-1] - agg.iloc[0]) * 100

    fig.add_trace(go.Bar(
        x=[f'D{d}' for d in agg.index], y=agg.values * 100,
        marker_color=colors, text=[f'{v*100:.2f}%' for v in agg.values],
        textposition='outside', name='Avg Fwd Return',
    ), row=1, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=1, col=1)
    fig.add_annotation(
        x=0.5, y=1.05, xref='x domain', yref='y domain',
        text=f'Monotonicity: {mono:.3f} | Spread D10-D1: {spread:.2f}%',
        showarrow=False, font=dict(size=12, color='#FFD700'), row=1, col=1,
    )

    # 2. Feature importance
    imp_df = pd.DataFrame(importance[:25]).iloc[::-1]
    imp_colors = []
    for name in imp_df['name']:
        if 'poly' in name or 'pullback' in name: imp_colors.append('#FF5722')
        elif 'ret_' in name or 'mom' in name: imp_colors.append('#2196F3')
        elif any(x in name for x in ['amihud', 'spread', 'turnover']): imp_colors.append('#FF9800')
        elif any(x in name for x in ['rsi', 'macd', 'bb_', 'stoch', 'cci', 'donchian', 'ma_ratio']): imp_colors.append('#9C27B0')
        elif 'rvol' in name or 'vol_of' in name or 'skew' in name or 'kurt' in name: imp_colors.append('#E91E63')
        else: imp_colors.append('#4CAF50')
    fig.add_trace(go.Bar(
        y=imp_df['name'], x=imp_df['weight_signed'], orientation='h',
        marker_color=imp_colors, name='Weight (signed)',
    ), row=1, col=2)
    fig.add_vline(x=0, line_color='gray', line_dash='dot', row=1, col=2)

    # 3. IC over time
    ic = stats_df['rank_ic']
    ic_roll = ic.rolling(6, min_periods=1).mean()
    avg_ic = ic.mean()
    ic_pos = (ic > 0).mean() * 100
    fig.add_trace(go.Bar(
        x=stats_df['date_str'], y=ic,
        marker_color=['#4CAF50' if v > 0 else '#f44336' for v in ic],
        opacity=0.4, name='Rank IC',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=stats_df['date_str'], y=ic_roll,
        line=dict(color='#FFD700', width=2.5), name='IC 6-MA',
    ), row=2, col=1)
    fig.add_hline(y=0, line_color='gray', row=2, col=1)
    fig.add_annotation(
        x=0.5, y=1.05, xref='x3 domain', yref='y3 domain',
        text=f'Avg IC: {avg_ic:.4f} | IC>0: {ic_pos:.0f}% | IR: {avg_ic/(ic.std()+1e-10):.2f}',
        showarrow=False, font=dict(size=12, color='#FFD700'), row=2, col=1,
    )

    # 4. IC distribution
    fig.add_trace(go.Histogram(x=ic, nbinsx=30, marker_color='#2196F3', opacity=0.7), row=2, col=2)
    fig.add_vline(x=0, line_color='red', line_dash='dot', row=2, col=2)
    fig.add_vline(x=avg_ic, line_color='#FFD700', row=2, col=2)

    # 5. Heatmap
    pivot = records_df.groupby(['date_str', 'decile'])['mean_fwd_ret'].mean().unstack(fill_value=0) * 100
    fig.add_trace(go.Heatmap(
        z=pivot.values.T, x=list(pivot.index), y=[f'D{d}' for d in pivot.columns],
        colorscale='RdYlGn', zmid=0, colorbar=dict(title='Ret%', len=0.2, y=0.5),
    ), row=3, col=1)

    # 6. Cumulative spread
    cum_spread = stats_df['spread'].cumsum() * 100
    fig.add_trace(go.Scatter(
        x=stats_df['date_str'], y=cum_spread, fill='tozeroy',
        line=dict(color='#03A9F4', width=2), name='Cum. Spread',
    ), row=3, col=2)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=3, col=2)

    # 7. Top vs Bottom
    fig.add_trace(go.Scatter(
        x=stats_df['date_str'], y=stats_df['top_decile_ret']*100,
        name='D10', line=dict(color='#4CAF50', width=2),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=stats_df['date_str'], y=stats_df['bot_decile_ret']*100,
        name='D1', line=dict(color='#f44336', width=2),
    ), row=4, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=4, col=1)

    # 8. Box plot
    for d in range(1, N_DECILES+1):
        sub = records_df[records_df['decile']==d]['mean_fwd_ret']*100
        fig.add_trace(go.Box(y=sub, name=f'D{d}', marker_color=colors[d-1], boxmean='sd', showlegend=False), row=4, col=2)

    # 9. Hit rate
    hr = records_df.groupby('decile')['mean_fwd_ret'].apply(lambda x: (x > 0).mean() * 100)
    fig.add_trace(go.Bar(
        x=[f'D{d}' for d in hr.index], y=hr.values, marker_color=colors[:len(hr)],
        text=[f'{v:.0f}%' for v in hr], textposition='outside', showlegend=False,
    ), row=5, col=1)
    fig.add_hline(y=50, line_color='gray', line_dash='dot', row=5, col=1)

    # 10. Sharpe
    ds = records_df.groupby('decile')['mean_fwd_ret'].apply(lambda x: x.mean() / (x.std()+1e-10))
    fig.add_trace(go.Bar(
        x=[f'D{d}' for d in ds.index], y=ds.values, marker_color=colors[:len(ds)],
        text=[f'{v:.3f}' for v in ds], textposition='outside', showlegend=False,
    ), row=5, col=2)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=5, col=2)

    avg_spread = stats_df['spread'].mean() * 100
    fig.update_layout(
        height=2600, width=1400, template='plotly_dark',
        title_text=(
            f'Cross-Factor Decile Calibration v2 — Pairwise Ranker (Numba JIT)<br>'
            f'<sub>Avg IC: {avg_ic:.4f} | Spread D10-D1: {spread:.2f}%/period | '
            f'Mono: {mono:.3f} | {len(stats_df)} periods ({HOLDING_DAYS}d hold) | '
            f'Excess return labels | Tail-weighted pairs | Near-tie filtered | {elapsed:.1f}s</sub>'
        ),
        showlegend=True, legend=dict(orientation='h', y=-0.02),
    )

    path = os.path.join(RESULTS_DIR, 'decile_calibration_v2.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")
    return path


def main():
    os.chdir(PROJECT_DIR)

    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    print(f"Features: {len(feat_cols)}")

    records_df, stats_df, importance, elapsed = run_walk_forward(panel, feat_cols)

    if records_df.empty:
        print("No results!")
        return

    # Save
    records_df.to_parquet(os.path.join(RESULTS_DIR, 'decile_v2_records.parquet'))
    stats_df.to_parquet(os.path.join(RESULTS_DIR, 'decile_v2_stats.parquet'))

    path = build_dashboard(records_df, stats_df, importance, elapsed)

    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
