"""
Elastic Net Grid Search for Pairwise Ranker
=============================================
Sweeps L1 and L2 regularization to find the sweet spot.
Uses the same pairwise ranking engine from decile_calibration_v2.
Reports: monotonicity, spread, IC, sparsity for each (L1, L2) pair.
"""
import numpy as np
import pandas as pd
import os
import sys
import time
import warnings
import itertools
warnings.filterwarnings('ignore')

# Reuse everything from v2
from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _sample_pairs_and_train_epoch,
    _train_model, _predict, _spearman_corr,
    TRAIN_DAYS, PURGE_DAYS, HOLDING_DAYS, REBAL_EVERY, MIN_COINS,
    N_DECILES, VOL_FLOOR_PCT, N_EPOCHS, PAIRS_PER_DATE,
    NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, RETRAIN_EVERY,
)
from scipy.stats import spearmanr

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ─── Grid (compact: 20 combos, ~2h) ───
L1_GRID = [0.0, 0.005, 0.02, 0.05, 0.1]
L2_GRID = [0.005, 0.01, 0.05, 0.1]
SWEEP_EPOCHS = 40  # fewer epochs for sweep (half of production)


def run_single_config(date_groups, sorted_dates, n_feat, l1, l2):
    """Run walk-forward for a single (L1, L2) config. Returns summary dict."""
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        days_from_start = (d - sorted_dates[0]).days
        if days_from_start >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i
            break

    rebal_dates = sorted_dates[start_idx::REBAL_EVERY]

    period_stats = []
    decile_means_all = []
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

            weights = _train_model(
                all_features, all_excess_ret, date_offsets,
                n_feat, SWEEP_EPOCHS, PAIRS_PER_DATE,
                NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, l1, l2,
                42 + ri,
            )
            last_train_ri = ri

        if last_train_ri is None:
            continue
        if pred_date not in date_groups:
            continue

        dg = date_groups[pred_date]
        scores = _predict(dg['features'], weights)
        n_coins = len(scores)

        rank_ic = float(_spearman_corr(scores, dg['excess_ret']))

        sorted_idx = np.argsort(scores)
        decile_sums = np.zeros(N_DECILES)
        decile_counts = np.zeros(N_DECILES, dtype=np.int64)
        for rank_pos, idx in enumerate(sorted_idx):
            decile = min(rank_pos * N_DECILES // n_coins, N_DECILES - 1)
            decile_sums[decile] += dg['fwd_ret'][idx]
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

    if not period_stats:
        return None

    stats = pd.DataFrame(period_stats)
    agg_decile = np.mean(decile_means_all, axis=0)
    mono, _ = spearmanr(range(N_DECILES), agg_decile)

    n_active = int(np.sum(np.abs(weights) > 1e-6))
    n_zero = n_feat - n_active

    return {
        'l1': l1,
        'l2': l2,
        'monotonicity': mono,
        'avg_ic': stats['rank_ic'].mean(),
        'ic_pos_pct': (stats['rank_ic'] > 0).mean() * 100,
        'avg_spread': stats['spread'].mean() * 100,
        'avg_top_ret': stats['top_ret'].mean() * 100,
        'avg_bot_ret': stats['bot_ret'].mean() * 100,
        'n_active': n_active,
        'n_zero': n_zero,
        'sparsity_pct': n_zero / n_feat * 100,
        'decile_means': [round(d * 100, 3) for d in agg_decile],
    }


def main():
    os.chdir(PROJECT_DIR)
    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    n_feat = len([f for f in feat_cols if f in panel.columns])
    print(f"Features: {n_feat}")

    fwd_col = f'fwd_ret_{HOLDING_DAYS}d'
    if fwd_col not in panel.columns:
        panel[fwd_col] = panel.groupby(level='symbol')['close'].transform(
            lambda x: x.pct_change(HOLDING_DAYS).shift(-HOLDING_DAYS)
        )

    # Pre-build date groups (shared across all configs)
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
        market_ret = np.nanmean(fwd_ret)
        excess_ret = fwd_ret - market_ret

        date_groups[date] = {
            'features': features,
            'fwd_ret': fwd_ret,
            'excess_ret': excess_ret,
        }

    sorted_dates = sorted(date_groups.keys())
    print(f"  {len(sorted_dates)} valid dates")

    # JIT warmup
    print("JIT warmup...")
    dummy_feat = np.random.randn(20, n_feat)
    dummy_eret = np.random.randn(20)
    dummy_w = np.zeros(n_feat)
    _sample_pairs_and_train_epoch(dummy_feat, dummy_eret, dummy_w, 10, 30.0, 1.0, 0.001, 0.005, 0.01, 42)

    # Grid search
    combos = list(itertools.product(L1_GRID, L2_GRID))
    print(f"\nGrid search: {len(combos)} combinations ({len(L1_GRID)} L1 x {len(L2_GRID)} L2)")
    print(f"{'L1':>8} {'L2':>8} {'Mono':>8} {'IC':>8} {'Spread':>8} {'Active':>8} {'Spars%':>8} | Decile Returns", flush=True)
    print("-" * 120, flush=True)

    results = []
    t0 = time.time()

    for i, (l1, l2) in enumerate(combos):
        t1 = time.time()
        res = run_single_config(date_groups, sorted_dates, n_feat, l1, l2)
        dt = time.time() - t1

        if res is None:
            print(f"  L1={l1:.3f} L2={l2:.3f} — FAILED")
            continue

        results.append(res)
        decile_str = ' '.join(f'{d:6.2f}' for d in res['decile_means'])
        print(f"{l1:>8.3f} {l2:>8.3f} {res['monotonicity']:>8.3f} {res['avg_ic']:>8.4f} "
              f"{res['avg_spread']:>8.2f}% {res['n_active']:>6d}/{n_feat} "
              f"{res['sparsity_pct']:>7.1f}% | {decile_str}  [{dt:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Rank results
    df = pd.DataFrame(results)
    df['score'] = (
        df['monotonicity'].rank(pct=True) * 0.4 +
        df['avg_spread'].rank(pct=True) * 0.3 +
        df['avg_ic'].rank(pct=True) * 0.2 +
        df['sparsity_pct'].rank(pct=True) * 0.1
    )
    df = df.sort_values('score', ascending=False)

    print(f"\n{'='*80}")
    print(f"  TOP 10 CONFIGURATIONS (weighted score: 40% mono + 30% spread + 20% IC + 10% sparsity)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'L1':>8} {'L2':>8} {'Mono':>8} {'IC':>8} {'Spread':>8} {'Active':>8} {'Score':>8}")
    print(f"{'-'*80}")
    for rank, (_, row) in enumerate(df.head(10).iterrows()):
        print(f"{rank+1:>4} {row['l1']:>8.3f} {row['l2']:>8.3f} {row['monotonicity']:>8.3f} "
              f"{row['avg_ic']:>8.4f} {row['avg_spread']:>7.2f}% {int(row['n_active']):>6d} {row['score']:>8.3f}")

    print(f"\n  BEST: L1={df.iloc[0]['l1']:.3f}, L2={df.iloc[0]['l2']:.3f}")
    print(f"  Monotonicity: {df.iloc[0]['monotonicity']:.3f}")
    print(f"  Avg Spread: {df.iloc[0]['avg_spread']:.2f}%")
    print(f"  Avg IC: {df.iloc[0]['avg_ic']:.4f}")
    print(f"  Active features: {int(df.iloc[0]['n_active'])}/{n_feat}")
    print(f"  Decile means: {df.iloc[0]['decile_means']}")
    print(f"{'='*80}")

    # Save
    df.to_csv(os.path.join(RESULTS_DIR, 'elastic_net_grid.csv'), index=False)
    print(f"\nResults saved to results/elastic_net_grid.csv")

    return df


if __name__ == '__main__':
    main()
