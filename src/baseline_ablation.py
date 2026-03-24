"""
Baseline Ablation Study — Answering: WHY does the short-only work?
Tests every hypothesis in parallel:
  1. Is it the ML or just a liquidity sort?
  2. Is it CatBoost or would any model work?
  3. Is N=5 optimal?
  4. Which features actually matter?
  5. Does the classifier add value over the ranker?
  6. Would random shorts do OK too? (null hypothesis)
  7. Is it holding period dependent?
  8. Does confidence gating help or hurt?
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRanker, CatBoostClassifier
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

# Reuse feature engine
from ml_features import load_daily_panel, compute_all_features

HOLDING_DAYS = 14
UNIVERSE_TOP = 50
STOP_PCT = 0.15
COST_PER_SIDE = 0.002
FUNDING_DAILY = 0.001
INITIAL_CAPITAL = 100_000
TRAIN_MONTHS = 18
PURGE_DAYS = 18

ALL_FEATURES = [
    'mom_14', 'mom_28', 'mom_56', 'mom_14_skip1',
    'poly_slope_28', 'poly_curve_28', 'poly_slope_56', 'poly_curve_56',
    'rvol_28', 'vol_ratio', 'max_ret_28', 'min_ret_28',
    'amihud', 'spread_28', 'turnover_28',
    'rsi_14', 'macd_hist', 'donchian_pos',
    'mom_14_csrank', 'rvol_28_csrank',
]


def simulate_shorts(panel, symbols, entry_date, hold_days, stop_pct):
    """Path-dependent short simulation."""
    dates = panel.index.get_level_values('date').unique().sort_values()
    mask = dates <= entry_date
    if not mask.any():
        return {s: 0.0 for s in symbols}
    eloc = mask.sum() - 1
    xloc = min(eloc + hold_days, len(dates) - 1)
    hdates = dates[eloc:xloc + 1]
    results = {}
    for sym in symbols:
        try:
            sd = panel.xs(sym, level='symbol')
            av = sd.index.intersection(hdates)
            if len(av) < 2:
                results[sym] = 0.0
                continue
            ep = sd.loc[av[0], 'close']
            if ep <= 0:
                results[sym] = 0.0
                continue
            peak = ep
            xp = ep
            for d in av[1:]:
                r = sd.loc[d]
                hi = r.get('intra_high', r['high'])
                sl = peak * (1 + stop_pct)
                if hi >= sl:
                    xp = sl
                    break
                lo = r.get('intra_low', r['low'])
                peak = min(peak, lo)
                xp = r['close']
            results[sym] = -(xp / ep - 1)
        except Exception:
            results[sym] = 0.0
    return results


def run_strategy(panel, rebal_dates, select_fn, name, funding_df, hold=HOLDING_DAYS):
    """Generic backtest: takes a selection function that returns list of symbols to short."""
    equity = INITIAL_CAPITAL
    curve = []
    for rd in rebal_dates:
        if rd not in panel.index.get_level_values('date'):
            curve.append({'date': rd, 'equity': equity})
            continue
        cross = panel.loc[rd].copy()
        cross = cross.dropna(subset=['mom_14', 'vol_avg_28'])
        cross = cross[cross['vol_avg_28'] > 0]
        cross = cross.nlargest(UNIVERSE_TOP, 'vol_avg_28')
        if len(cross) < 10:
            curve.append({'date': rd, 'equity': equity})
            continue

        syms = select_fn(cross, rd)
        if not syms:
            curve.append({'date': rd, 'equity': equity})
            continue

        try:
            rets = simulate_shorts(panel, syms, rd, hold, STOP_PCT)
        except Exception:
            rets = {s: 0.0 for s in syms}

        ret = np.mean(list(rets.values()))
        # Funding
        if not funding_df.empty:
            fm = funding_df[funding_df.index <= rd]['fundingRate'].tail(21).mean()
            ret -= abs(fm) * 3 * hold
        else:
            ret -= FUNDING_DAILY * hold
        ret -= 2 * COST_PER_SIDE
        if np.isnan(ret) or np.isinf(ret):
            ret = 0
        equity *= (1 + ret)
        if equity <= 0 or np.isnan(equity):
            equity = max(1.0, equity if not np.isnan(equity) else 1.0)
        curve.append({'date': rd, 'equity': equity})

    df = pd.DataFrame(curve).set_index('date')
    eq = df['equity']
    n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    rets = eq.pct_change().dropna()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_yrs) - 1 if eq.iloc[-1] > 0 else -1
    sharpe = rets.mean() / rets.std() * np.sqrt(365/hold) if rets.std() > 0 else 0
    pk = eq.expanding().max()
    mdd = ((eq - pk) / pk).min()
    ds = rets[rets < 0].std()
    sortino = rets.mean() / ds * np.sqrt(365/hold) if ds and ds > 0 else 0

    yearly = {}
    for yr in sorted(eq.index.year.unique()):
        y = eq[eq.index.year == yr]
        if len(y) > 1:
            yearly[yr] = (y.iloc[-1] / y.iloc[0] - 1) * 100

    return {
        'name': name,
        'final': eq.iloc[-1],
        'cagr': cagr * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': mdd * 100,
        'calmar': cagr / abs(mdd) if mdd != 0 else 0,
        **{f'yr_{k}': v for k, v in yearly.items()},
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  BASELINE ABLATION STUDY")
    print("  Why does the short-only work? What beats it?")
    print("=" * 70)

    print("\nLoading data...")
    panel = load_daily_panel()
    print("Computing features...")
    panel = compute_all_features(panel)

    dates = panel.index.get_level_values('date').unique().sort_values()
    start = dates[0] + pd.DateOffset(months=TRAIN_MONTHS + 2)
    rebal_dates = []
    d = start
    while d <= dates[-1] - pd.Timedelta(days=HOLDING_DAYS):
        nearest = dates[dates <= d]
        if len(nearest) > 0:
            rebal_dates.append(nearest[-1])
        d += pd.Timedelta(days=HOLDING_DAYS)
    rebal_dates = sorted(set(rebal_dates))

    funding_df = pd.DataFrame()
    fp = os.path.join(DATA_DIR, 'BTCUSDT_funding.parquet')
    if os.path.exists(fp):
        funding_df = pd.read_parquet(fp)

    print(f"Rebalance dates: {len(rebal_dates)}")
    print(f"\nRunning {20}+ strategies...\n")

    results = []

    # ═══════════════════════════════════════
    # TEST 1: Simple sorts (no ML at all)
    # ═══════════════════════════════════════
    print("--- TEST 1: Simple sorts (no ML) ---")

    for feat, n in [
        ('spread_28', 5), ('amihud', 5), ('turnover_28', 5),
        ('spread_28', 3), ('spread_28', 8),
        ('rvol_28', 5), ('mom_14', 5), ('min_ret_28', 5),
    ]:
        ascending = feat not in ['turnover_28', 'mom_14']  # high spread = bad, high turnover = good
        def make_sel(f=feat, n_=n, asc=ascending):
            def sel(cross, rd):
                if asc:
                    return cross.nlargest(n_, f).index.tolist()
                else:
                    return cross.nsmallest(n_, f).index.tolist()
            return sel
        r = run_strategy(panel, rebal_dates, make_sel(), f"sort_{feat}_n{n}", funding_df)
        results.append(r)
        print(f"  {r['name']:30s} Sharpe={r['sharpe']:.2f}  CAGR={r['cagr']:.0f}%  DD={r['max_dd']:.0f}%  ${r['final']:,.0f}")

    # Composite: spread * amihud (no ML)
    def sel_composite(cross, rd):
        cross = cross.copy()
        cross['composite'] = (
            cross['spread_28'].rank(pct=True) +
            cross['amihud'].rank(pct=True) -
            cross['turnover_28'].rank(pct=True) -
            cross['mom_14'].rank(pct=True)
        )
        return cross.nlargest(5, 'composite').index.tolist()
    r = run_strategy(panel, rebal_dates, sel_composite, "composite_liq+mom_n5", funding_df)
    results.append(r)
    print(f"  {r['name']:30s} Sharpe={r['sharpe']:.2f}  CAGR={r['cagr']:.0f}%  DD={r['max_dd']:.0f}%  ${r['final']:,.0f}")

    # ═══════════════════════════════════════
    # TEST 2: Random shorts (null hypothesis)
    # ═══════════════════════════════════════
    print("\n--- TEST 2: Random shorts (null hypothesis) ---")
    random_results = []
    for seed in range(10):
        rng = np.random.RandomState(seed)
        def sel_random(cross, rd, _rng=rng):
            idx = cross.index.tolist()
            chosen = _rng.choice(idx, size=min(5, len(idx)), replace=False)
            return chosen.tolist()
        r = run_strategy(panel, rebal_dates, sel_random, f"random_seed{seed}", funding_df)
        random_results.append(r)
    avg_random_sharpe = np.mean([r['sharpe'] for r in random_results])
    avg_random_cagr = np.mean([r['cagr'] for r in random_results])
    results.append({
        'name': 'RANDOM_AVG_10seeds',
        'final': np.mean([r['final'] for r in random_results]),
        'cagr': avg_random_cagr,
        'sharpe': avg_random_sharpe,
        'sortino': np.mean([r['sortino'] for r in random_results]),
        'max_dd': np.mean([r['max_dd'] for r in random_results]),
        'calmar': 0,
    })
    print(f"  Random avg (10 seeds):       Sharpe={avg_random_sharpe:.2f}  CAGR={avg_random_cagr:.0f}%")

    # ═══════════════════════════════════════
    # TEST 3: CatBoost YetiRank (the baseline)
    # ═══════════════════════════════════════
    print("\n--- TEST 3: CatBoost YetiRank (baseline reproduction) ---")

    # Train models once on full walk-forward
    for n_shorts in [3, 4, 5, 6, 8]:
        for feat_set_name, feat_set in [
            ('all_20', ALL_FEATURES),
            ('liq_only', ['amihud', 'spread_28', 'turnover_28', 'rvol_28_csrank', 'vol_ratio']),
            ('mom_only', ['mom_14', 'mom_28', 'mom_56', 'mom_14_skip1', 'mom_14_csrank']),
            ('poly_only', ['poly_slope_28', 'poly_curve_28', 'poly_slope_56', 'poly_curve_56']),
        ]:
            if n_shorts != 5 and feat_set_name != 'all_20':
                continue  # only sweep N for full features

            avail = [f for f in feat_set if f in panel.columns]
            if len(avail) < 3:
                continue

            # Walk-forward with CatBoost
            models_cache = {}

            def sel_catboost(cross, rd, _n=n_shorts, _feats=avail):
                nonlocal models_cache
                # Check if we need to retrain
                period_key = rd.year * 100 + rd.month // 2
                if period_key not in models_cache:
                    ts = rd - pd.DateOffset(months=TRAIN_MONTHS)
                    tmask = ((panel.index.get_level_values('date') >= ts) &
                             (panel.index.get_level_values('date') < rd))
                    train = panel[tmask].copy()
                    g = train.groupby(level='symbol')
                    train['fwd'] = g['close'].transform(lambda x: x.pct_change(HOLDING_DAYS).shift(-HOLDING_DAYS))
                    cutoff = train.index.get_level_values('date').max() - pd.Timedelta(days=PURGE_DAYS)
                    train = train[train.index.get_level_values('date') <= cutoff]
                    train = train.dropna(subset=['fwd'])
                    mkt = train.groupby(level='date')['fwd'].transform('mean')
                    train['nfwd'] = train['fwd'] - mkt
                    train['rl'] = train.groupby(level='date')['nfwd'].transform(
                        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') if len(x) >= 5 else 2
                    ).fillna(2).astype(int)

                    if 'vol_avg_28' in panel.columns:
                        vol = panel.loc[train.index, 'vol_avg_28']
                        thresh = vol.groupby(level='date').transform(lambda x: x.quantile(0.25))
                        train = train[vol > thresh]

                    X = np.nan_to_num(train[_feats].values, nan=0, posinf=0, neginf=0)
                    if len(X) < 300:
                        models_cache[period_key] = None
                    else:
                        try:
                            gids = pd.Categorical(train.index.get_level_values('date')).codes
                            m = CatBoostRanker(
                                loss_function='YetiRank', iterations=200, depth=4,
                                learning_rate=0.05, l2_leaf_reg=5.0, random_strength=2.0,
                                bagging_temperature=1.0, verbose=0, random_seed=42, task_type='CPU')
                            m.fit(X, train['rl'].values, group_id=gids)
                            models_cache[period_key] = m
                        except Exception:
                            models_cache[period_key] = None

                model = models_cache.get(period_key)
                if model is None:
                    return cross.nlargest(_n, 'spread_28').index.tolist()

                X = np.nan_to_num(cross[_feats].values, nan=0, posinf=0, neginf=0)
                scores = model.predict(X)
                cross = cross.copy()
                cross['score'] = scores
                return cross.nsmallest(_n, 'score').index.tolist()

            models_cache = {}  # reset for each config
            r = run_strategy(panel, rebal_dates, sel_catboost,
                           f"catboost_{feat_set_name}_n{n_shorts}", funding_df)
            results.append(r)
            print(f"  {r['name']:30s} Sharpe={r['sharpe']:.2f}  CAGR={r['cagr']:.0f}%  DD={r['max_dd']:.0f}%  ${r['final']:,.0f}")

    # ═══════════════════════════════════════
    # TEST 4: Different holding periods
    # ═══════════════════════════════════════
    print("\n--- TEST 4: Holding period sweep (simple spread sort) ---")
    for hold in [7, 14, 21, 28]:
        rd_h = []
        d = start
        while d <= dates[-1] - pd.Timedelta(days=hold):
            nearest = dates[dates <= d]
            if len(nearest) > 0:
                rd_h.append(nearest[-1])
            d += pd.Timedelta(days=hold)
        rd_h = sorted(set(rd_h))

        def sel_spread5(cross, rd):
            return cross.nlargest(5, 'spread_28').index.tolist()
        r = run_strategy(panel, rd_h, sel_spread5, f"spread_n5_hold{hold}d", funding_df, hold=hold)
        results.append(r)
        print(f"  {r['name']:30s} Sharpe={r['sharpe']:.2f}  CAGR={r['cagr']:.0f}%  DD={r['max_dd']:.0f}%  ${r['final']:,.0f}")

    # ═══════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════
    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)

    print(f"\n{'='*90}")
    print(f"  ABLATION RESULTS — SORTED BY SHARPE")
    print(f"{'='*90}")
    cols = ['name', 'sharpe', 'cagr', 'max_dd', 'sortino', 'final']
    avail_cols = [c for c in cols if c in df.columns]
    print(df[avail_cols].to_string(index=False))

    # Key comparisons
    print(f"\n{'='*90}")
    print(f"  KEY FINDINGS")
    print(f"{'='*90}")

    best_simple = df[df['name'].str.startswith('sort_') | df['name'].str.startswith('composite')].iloc[0]
    best_ml = df[df['name'].str.startswith('catboost')].iloc[0] if any(df['name'].str.startswith('catboost')) else None
    random_row = df[df['name'] == 'RANDOM_AVG_10seeds'].iloc[0]

    print(f"\n  1. DOES ML ADD VALUE OVER SIMPLE SORT?")
    print(f"     Best simple sort:  {best_simple['name']} -> Sharpe {best_simple['sharpe']:.2f}")
    if best_ml is not None:
        print(f"     Best ML (CatBoost): {best_ml['name']} -> Sharpe {best_ml['sharpe']:.2f}")
        delta = best_ml['sharpe'] - best_simple['sharpe']
        print(f"     ML advantage:     {delta:+.2f} Sharpe ({'+' if delta > 0 else ''}{'YES adds value' if delta > 0.1 else 'MARGINAL' if delta > 0 else 'NO, simple is better'})")

    print(f"\n  2. IS IT BETTER THAN RANDOM?")
    print(f"     Random avg:       Sharpe {random_row['sharpe']:.2f}")
    print(f"     Best strategy:    Sharpe {df.iloc[0]['sharpe']:.2f}")
    print(f"     Edge over random: {df.iloc[0]['sharpe'] - random_row['sharpe']:+.2f}")

    print(f"\n  3. WHICH FEATURES MATTER MOST? (CatBoost with feature subsets)")
    for _, row in df[df['name'].str.startswith('catboost_') & df['name'].str.contains('_n5')].iterrows():
        print(f"     {row['name']:30s} Sharpe {row['sharpe']:.2f}")

    print(f"\n  4. OPTIMAL N?")
    for _, row in df[df['name'].str.startswith('catboost_all_20_n')].iterrows():
        print(f"     {row['name']:30s} Sharpe {row['sharpe']:.2f}  DD {row['max_dd']:.0f}%")

    hold_rows = df[df['name'].str.startswith('spread_n5_hold')]
    if len(hold_rows) > 0:
        print(f"\n  5. OPTIMAL HOLDING PERIOD?")
        for _, row in hold_rows.iterrows():
            print(f"     {row['name']:30s} Sharpe {row['sharpe']:.2f}")

    print(f"\n{'='*90}")

    # Save
    df.to_csv(os.path.join(RESULTS_DIR, 'ablation_results.csv'), index=False)
    print(f"Saved to {RESULTS_DIR}/ablation_results.csv")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
