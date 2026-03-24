"""
Rolling Window Grid Search — Parallel
=======================================
Sweeps training window, retrain frequency, rebalance interval,
purge gap, and holding period. No burn-in waste.

Parallelized via ProcessPoolExecutor (12 cores).
Fixed: L1=0, L2=0.01 (from elastic net grid).
"""
import numpy as np
import pandas as pd
import os
import sys
import time
import warnings
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# Fixed params from elastic net grid
L1_REG = 0.0
L2_REG = 0.01
LR = 0.002
N_EPOCHS = 50           # slightly reduced for speed in grid
PAIRS_PER_DATE = 500
NEAR_TIE_PCT = 30.0
TAIL_WEIGHT_POW = 1.0
MIN_COINS = 20
N_DECILES = 10
VOL_FLOOR_PCT = 0.20
N_WORKERS = 10           # parallel workers (leave 2 cores free)

# ─── Grid ───
GRID = {
    'train_days':    [180, 270, 365, 540],
    'retrain_every': [1, 2, 4, 8],        # in rebalance periods
    'rebal_days':    [7, 14, 28],
    'purge_days':    [7, 14, 21],
    'hold_days':     [7, 14, 28],
}

# ════════════════════════════════════════════
# NUMBA KERNELS (copied to be pickle-safe for multiprocessing)
# ════════════════════════════════════════════
from numba import njit

@njit(cache=True)
def _rankdata(x):
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n)
    for i in range(n):
        ranks[order[i]] = i + 1.0
    return ranks

@njit(cache=True)
def _cross_sectional_rank_normalize(features):
    n, p = features.shape
    out = np.empty_like(features)
    for j in range(p):
        ranks = _rankdata(features[:, j])
        for i in range(n):
            out[i, j] = ranks[i] / n
    return out

@njit(cache=True)
def _sigmoid(x):
    if x > 20.0: return 1.0
    elif x < -20.0: return 0.0
    return 1.0 / (1.0 + np.exp(-x))

@njit(cache=True)
def _percentile(arr, pct):
    s = np.sort(arr)
    idx = int(pct / 100.0 * (len(s) - 1) + 0.5)
    return s[min(idx, len(s) - 1)]

@njit(cache=True)
def _sample_pairs_and_train_epoch(
    features, excess_ret, weights, n_pairs,
    near_tie_pct, tail_pow, lr, l1_reg, l2_reg, rng_seed,
):
    n, p = features.shape
    if n < 4: return 0.0, 0
    scores = features @ weights
    sorted_idx = np.argsort(excess_ret)
    tail_size = max(2, int(n * 0.2))
    top_start = n - tail_size
    seed = rng_seed
    n_diff_samples = min(500, n * (n - 1) // 2)
    diffs = np.empty(n_diff_samples)
    for k in range(n_diff_samples):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        i = int(seed % n)
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        j = int(seed % n)
        if i == j: j = (j + 1) % n
        diffs[k] = abs(excess_ret[i] - excess_ret[j])
    epsilon = _percentile(diffs, near_tie_pct)
    grad = np.zeros(p)
    total_loss = 0.0
    valid_pairs = 0
    for k in range(n_pairs):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        if (seed >> 32) % 2 == 0:
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            i = sorted_idx[top_start + int(seed % tail_size)]
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            j = sorted_idx[int(seed % tail_size)]
        else:
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            i = int(seed % n)
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            j = int(seed % n)
        if i == j: continue
        abs_diff = abs(excess_ret[i] - excess_ret[j])
        if abs_diff < epsilon: continue
        w = abs_diff ** tail_pow
        sigma = 1.0 if excess_ret[i] > excess_ret[j] else -1.0
        diff = scores[i] - scores[j]
        margin = sigma * diff
        if margin > 20.0: loss = 0.0
        elif margin < -20.0: loss = -margin
        else: loss = np.log(1.0 + np.exp(-margin))
        total_loss += w * loss
        sig = _sigmoid(-margin)
        coeff = -w * sigma * sig
        for f in range(p):
            grad[f] += coeff * (features[i, f] - features[j, f])
        valid_pairs += 1
    if valid_pairs > 0:
        scale = 1.0 / valid_pairs
        for f in range(p):
            l1_grad = l1_reg * (1.0 if weights[f] > 0 else (-1.0 if weights[f] < 0 else 0.0))
            g = grad[f] * scale + 2.0 * l2_reg * weights[f] + l1_grad
            weights[f] -= lr * g
    return total_loss, valid_pairs

@njit(cache=True)
def _train_model(all_features, all_excess_ret, date_offsets, n_feat,
                 n_epochs, n_pairs, near_tie_pct, tail_pow, lr, l1_reg, l2_reg, seed):
    weights = np.zeros(n_feat)
    n_dates = len(date_offsets) - 1
    for epoch in range(n_epochs):
        for d in range(n_dates):
            start = date_offsets[d]
            end = date_offsets[d + 1]
            if end - start < 5: continue
            _sample_pairs_and_train_epoch(
                all_features[start:end], all_excess_ret[start:end], weights,
                n_pairs, near_tie_pct, tail_pow, lr, l1_reg, l2_reg,
                seed + epoch * 1000 + d,
            )
    return weights

@njit(cache=True)
def _spearman_corr(x, y):
    rx = _rankdata(x)
    ry = _rankdata(y)
    n = len(x)
    mx, my = 0.0, 0.0
    for i in range(n): mx += rx[i]; my += ry[i]
    mx /= n; my /= n
    cov, vx, vy = 0.0, 0.0, 0.0
    for i in range(n):
        dx = rx[i] - mx; dy = ry[i] - my
        cov += dx * dy; vx += dx * dx; vy += dy * dy
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 1e-15 else 0.0


# ════════════════════════════════════════════
# DATA LOADING (shared across workers)
# ════════════════════════════════════════════

def load_panel_data():
    """Load panel and precompute per-date arrays for all possible hold_days."""
    panel_path = os.path.join(DATA_DIR, 'feature_panel_v2.parquet')
    print("Loading feature_panel_v2.parquet...")
    panel = pd.read_parquet(panel_path)
    print(f"  Panel: {panel.shape}")

    # Feature columns
    exclude_prefixes = ('fwd_ret_', 'fwd_')
    exclude_exact = {
        'open', 'high', 'low', 'close', 'volume', 'quote_vol',
        'log_ret', 'spread_proxy', 'atr_14d', 'symbol',
    }
    feat_cols = [c for c in panel.columns
                 if c not in exclude_exact
                 and not any(c.startswith(p) for p in exclude_prefixes)
                 and panel[c].dtype in ('float64', 'float32', 'int64')]
    n_feat = len(feat_cols)
    print(f"  Features: {n_feat}")

    # Ensure forward return columns exist for all hold_days
    for hd in GRID['hold_days']:
        col = f'fwd_ret_{hd}d'
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(hd).shift(-hd)
            )

    # Group by date, precompute feature matrices
    print("Building date groups...")
    date_groups = {}
    for date, group in panel.groupby(level='date'):
        # Base filter (close exists)
        g = group.dropna(subset=['close'])
        if len(g) < MIN_COINS:
            continue

        # Volume filter
        if 'turnover_28d' in g.columns:
            thresh = g['turnover_28d'].quantile(VOL_FLOOR_PCT)
            g = g[g['turnover_28d'] >= thresh]
        if len(g) < MIN_COINS:
            continue

        # Features (rank-normalized)
        available = [f for f in feat_cols if f in g.columns]
        features = g[available].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = _cross_sectional_rank_normalize(features)

        # Forward returns for each holding period
        fwd_rets = {}
        for hd in GRID['hold_days']:
            col = f'fwd_ret_{hd}d'
            if col in g.columns:
                fr = g[col].values.astype(np.float64)
                fr = np.nan_to_num(fr, nan=0.0)
                fwd_rets[hd] = fr

        date_groups[date] = {
            'features': features,
            'fwd_rets': fwd_rets,
        }

    sorted_dates = sorted(date_groups.keys())
    print(f"  {len(sorted_dates)} valid dates, {n_feat} features")
    return date_groups, sorted_dates, n_feat


# ════════════════════════════════════════════
# SINGLE CONFIG RUNNER
# ════════════════════════════════════════════

def run_config(args):
    """Run a single config. Returns result dict."""
    (config_id, train_days, retrain_every, rebal_days, purge_days, hold_days,
     date_groups_path, sorted_dates_path, n_feat) = args

    # Load shared data from disk (serialized for multiprocessing)
    import pickle
    with open(date_groups_path, 'rb') as f:
        date_groups = pickle.load(f)
    with open(sorted_dates_path, 'rb') as f:
        sorted_dates = pickle.load(f)

    t0 = time.time()

    # Build rebalance schedule (no burn-in waste — start as soon as we have train_days)
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if (d - sorted_dates[0]).days >= train_days + purge_days:
            start_idx = i
            break

    rebal_indices = list(range(start_idx, len(sorted_dates), max(1, rebal_days)))

    if len(rebal_indices) < 5:
        return None

    period_stats = []
    decile_means_all = []
    weights = np.zeros(n_feat)
    last_train_ri = None

    for ri, cs_idx in enumerate(rebal_indices):
        pred_date = sorted_dates[cs_idx]
        should_train = (last_train_ri is None or ri % retrain_every == 0)

        if should_train:
            train_end = pred_date - pd.Timedelta(days=purge_days)
            train_start = train_end - pd.Timedelta(days=train_days)

            # Collect training dates (rolling window, not expanding)
            train_dates = [d for d in sorted_dates
                          if train_start <= d <= train_end and d in date_groups]
            if len(train_dates) < 10:
                continue

            # Check if this hold_days has fwd data
            valid_train = []
            for td in train_dates:
                dg = date_groups[td]
                if hold_days in dg['fwd_rets']:
                    fr = dg['fwd_rets'][hold_days]
                    if np.sum(np.isfinite(fr) & (fr != 0)) >= MIN_COINS * 0.5:
                        valid_train.append(td)
            if len(valid_train) < 10:
                continue

            # Concatenate
            all_feat_list = []
            all_eret_list = []
            offsets = [0]
            for td in valid_train:
                dg = date_groups[td]
                fr = dg['fwd_rets'][hold_days]
                mask = np.isfinite(fr) & (fr != 0)
                if np.sum(mask) < 5:
                    continue
                feat = dg['features'][mask]
                fwd = fr[mask]
                excess = fwd - np.mean(fwd)
                all_feat_list.append(feat)
                all_eret_list.append(excess)
                offsets.append(offsets[-1] + len(excess))

            if len(all_feat_list) < 5:
                continue

            all_features = np.vstack(all_feat_list)
            all_excess_ret = np.concatenate(all_eret_list)
            date_offsets = np.array(offsets, dtype=np.int64)

            weights = _train_model(
                all_features, all_excess_ret, date_offsets,
                n_feat, N_EPOCHS, PAIRS_PER_DATE,
                NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG,
                42 + config_id * 1000 + ri,
            )
            last_train_ri = ri

        if last_train_ri is None:
            continue
        if pred_date not in date_groups:
            continue

        dg = date_groups[pred_date]
        if hold_days not in dg['fwd_rets']:
            continue

        fwd_ret = dg['fwd_rets'][hold_days]
        mask = np.isfinite(fwd_ret) & (fwd_ret != 0)
        if np.sum(mask) < MIN_COINS:
            continue

        features = dg['features'][mask]
        fwd = fwd_ret[mask]
        excess = fwd - np.mean(fwd)
        scores = features @ weights
        n_coins = len(scores)

        rank_ic = float(_spearman_corr(scores, excess))

        sorted_idx = np.argsort(scores)
        decile_sums = np.zeros(N_DECILES)
        decile_counts = np.zeros(N_DECILES, dtype=np.int64)
        for rank_pos, idx in enumerate(sorted_idx):
            decile = min(rank_pos * N_DECILES // n_coins, N_DECILES - 1)
            decile_sums[decile] += fwd[idx]
            decile_counts[decile] += 1

        decile_means = np.where(decile_counts > 0, decile_sums / decile_counts, 0.0)
        top_ret = decile_means[N_DECILES - 1]
        bot_ret = decile_means[0]

        period_stats.append({
            'rank_ic': rank_ic,
            'spread': top_ret - bot_ret,
            'top_ret': top_ret,
            'bot_ret': bot_ret,
        })
        decile_means_all.append(decile_means)

    elapsed = time.time() - t0

    if len(period_stats) < 5:
        return None

    stats = pd.DataFrame(period_stats)
    agg_decile = np.mean(decile_means_all, axis=0)

    from scipy.stats import spearmanr
    mono, _ = spearmanr(range(N_DECILES), agg_decile)

    return {
        'config_id': config_id,
        'train_days': train_days,
        'retrain_every': retrain_every,
        'rebal_days': rebal_days,
        'purge_days': purge_days,
        'hold_days': hold_days,
        'monotonicity': mono,
        'avg_ic': stats['rank_ic'].mean(),
        'ic_pos_pct': (stats['rank_ic'] > 0).mean() * 100,
        'avg_spread': stats['spread'].mean() * 100,
        'avg_top_ret': stats['top_ret'].mean() * 100,
        'avg_bot_ret': stats['bot_ret'].mean() * 100,
        'n_periods': len(stats),
        'elapsed': elapsed,
        'decile_means': [round(d * 100, 3) for d in agg_decile],
    }


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

def main():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    # Load data once
    date_groups, sorted_dates, n_feat = load_panel_data()

    # Serialize shared data for multiprocessing
    import pickle, tempfile
    tmp_dir = tempfile.mkdtemp()
    dg_path = os.path.join(tmp_dir, 'date_groups.pkl')
    sd_path = os.path.join(tmp_dir, 'sorted_dates.pkl')
    print("Serializing shared data for workers...")
    with open(dg_path, 'wb') as f:
        pickle.dump(date_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(sd_path, 'wb') as f:
        pickle.dump(sorted_dates, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Serialized ({os.path.getsize(dg_path) / 1e6:.0f}MB)")

    # JIT warmup
    print("JIT warmup...")
    dummy_feat = np.random.randn(20, n_feat)
    dummy_eret = np.random.randn(20)
    dummy_w = np.zeros(n_feat)
    _sample_pairs_and_train_epoch(dummy_feat, dummy_eret, dummy_w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # Build config list — prune combos where hold > rebal (doesn't make sense)
    configs = []
    config_id = 0
    for td, rt, rb, pg, hd in itertools.product(
        GRID['train_days'], GRID['retrain_every'],
        GRID['rebal_days'], GRID['purge_days'], GRID['hold_days']
    ):
        # Skip nonsensical combos
        if hd > rb * 4:   # holding much longer than rebalance makes no sense
            continue
        if pg > td // 4:   # purge > 25% of training window is too much
            continue
        configs.append((config_id, td, rt, rb, pg, hd, dg_path, sd_path, n_feat))
        config_id += 1

    n_configs = len(configs)
    print(f"\nGrid: {n_configs} valid configurations")
    print(f"Workers: {N_WORKERS}")
    print(f"Estimated time: ~{n_configs * 120 / N_WORKERS / 60:.0f} min")
    print(f"\n{'ID':>4} {'Train':>6} {'Retr':>5} {'Rebal':>6} {'Purge':>6} {'Hold':>5} "
          f"{'Mono':>7} {'IC':>7} {'Spread':>8} {'#Per':>5} {'Time':>6}", flush=True)
    print("-" * 90, flush=True)

    # Run in parallel
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_config = {executor.submit(run_config, cfg): cfg for cfg in configs}

        for future in as_completed(future_to_config):
            cfg = future_to_config[future]
            try:
                res = future.result(timeout=600)
                if res is None:
                    continue
                results.append(res)
                print(f"{res['config_id']:>4} {res['train_days']:>6} {res['retrain_every']:>5} "
                      f"{res['rebal_days']:>6} {res['purge_days']:>6} {res['hold_days']:>5} "
                      f"{res['monotonicity']:>7.3f} {res['avg_ic']:>7.4f} "
                      f"{res['avg_spread']:>7.2f}% {res['n_periods']:>5} "
                      f"{res['elapsed']:>5.0f}s", flush=True)
            except Exception as e:
                print(f"  Config {cfg[0]} failed: {e}", flush=True)

    # Cleanup
    os.unlink(dg_path)
    os.unlink(sd_path)
    os.rmdir(tmp_dir)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time / 60:.1f} min ({len(results)} successful configs)")

    if not results:
        print("No results!")
        return

    # Rank
    df = pd.DataFrame(results)
    df['score'] = (
        df['monotonicity'].rank(pct=True) * 0.35 +
        df['avg_spread'].rank(pct=True) * 0.30 +
        df['avg_ic'].rank(pct=True) * 0.20 +
        df['n_periods'].rank(pct=True) * 0.15
    )
    df = df.sort_values('score', ascending=False)

    print(f"\n{'='*100}")
    print(f"  TOP 15 CONFIGURATIONS")
    print(f"  (Score = 35% mono + 30% spread + 20% IC + 15% #periods)")
    print(f"{'='*100}")
    print(f"{'Rk':>3} {'Train':>6} {'Retr':>5} {'Rebal':>6} {'Purge':>6} {'Hold':>5} "
          f"{'Mono':>7} {'IC':>7} {'Spread':>8} {'IC>0':>6} {'#Per':>5} {'Score':>6}")
    print(f"{'-'*100}")
    for rank, (_, row) in enumerate(df.head(15).iterrows()):
        print(f"{rank+1:>3} {int(row['train_days']):>6} {int(row['retrain_every']):>5} "
              f"{int(row['rebal_days']):>6} {int(row['purge_days']):>6} {int(row['hold_days']):>5} "
              f"{row['monotonicity']:>7.3f} {row['avg_ic']:>7.4f} "
              f"{row['avg_spread']:>7.2f}% {row['ic_pos_pct']:>5.0f}% {int(row['n_periods']):>5} "
              f"{row['score']:>6.3f}")

    best = df.iloc[0]
    print(f"\n  BEST CONFIG:")
    print(f"    Train window: {int(best['train_days'])} days")
    print(f"    Retrain every: {int(best['retrain_every'])} rebalances")
    print(f"    Rebalance every: {int(best['rebal_days'])} days")
    print(f"    Purge gap: {int(best['purge_days'])} days")
    print(f"    Holding period: {int(best['hold_days'])} days")
    print(f"    Monotonicity: {best['monotonicity']:.3f}")
    print(f"    Avg Spread: {best['avg_spread']:.2f}%")
    print(f"    Avg IC: {best['avg_ic']:.4f}")
    print(f"    Deciles: {best['decile_means']}")
    print(f"{'='*100}")

    # Marginal analysis: best value per parameter
    print(f"\n  MARGINAL ANALYSIS (mean score by parameter value):")
    for param in ['train_days', 'retrain_every', 'rebal_days', 'purge_days', 'hold_days']:
        marginal = df.groupby(param)['score'].mean().sort_values(ascending=False)
        print(f"\n  {param}:")
        for val, score in marginal.items():
            n = len(df[df[param] == val])
            mono = df[df[param] == val]['monotonicity'].mean()
            spread = df[df[param] == val]['avg_spread'].mean()
            print(f"    {int(val):>6}: score={score:.3f}  mono={mono:.3f}  spread={spread:.2f}%  (n={n})")

    # Save
    df.to_csv(os.path.join(RESULTS_DIR, 'window_grid.csv'), index=False)
    print(f"\nSaved to results/window_grid.csv")

    return df


if __name__ == '__main__':
    main()
