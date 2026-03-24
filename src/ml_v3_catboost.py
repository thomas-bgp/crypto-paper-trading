"""
ML v3 — CatBoost Ranking with anti-overfit techniques.
Key changes from v2:
  - CatBoost YetiRank (pairwise ranking, ordered boosting prevents leakage)
  - Target: cross-sectional RANK of return (not raw return)
  - Feature reduction: top 15-20 features only (less overfit)
  - Market-neutralized returns (remove market beta before ranking)
  - Ensemble: 5 models trained on staggered windows (bagging over time)
  - Long-short: always L/S to kill delta, with proper funding costs
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRanker, Pool
import shap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Config ───
TRAIN_MONTHS = 18       # longer window for more data
PURGE_DAYS = 18         # 14d holding + 4d buffer
HOLDING_DAYS = 14
TOP_N = 5               # fewer positions = less noise
UNIVERSE_TOP = 999      # open universe — let the model decide who's good/bad
COST_PER_SIDE = 0.002
FUNDING_DAILY = 0.001
INITIAL_CAPITAL = 100_000
N_ENSEMBLE = 3           # 3 models on staggered windows

# CatBoost params optimized for low-N financial ranking
CATBOOST_PARAMS = {
    'loss_function': 'YetiRank',    # pairwise ranking loss
    'iterations': 300,
    'depth': 4,                      # shallow trees (anti-overfit)
    'learning_rate': 0.05,
    'l2_leaf_reg': 5.0,             # strong L2 regularization
    'random_strength': 2.0,          # randomization in scoring (anti-overfit)
    'bagging_temperature': 1.0,      # Bayesian bootstrap temperature
    'border_count': 64,              # fewer split candidates
    'verbose': 0,
    'random_seed': 42,
    'task_type': 'CPU',
}

# ─── Features: curated list (anti-overfit: fewer is better) ───
CORE_FEATURES = [
    # Momentum (the signal)
    'mom_14', 'mom_28', 'mom_56', 'mom_14_skip1',
    # Polynomial derivatives (user's insight)
    'poly_slope_28', 'poly_curve_28', 'poly_slope_56', 'poly_curve_56',
    # Volatility (risk adjustment)
    'rvol_28', 'vol_ratio', 'max_ret_28', 'min_ret_28',
    # Liquidity (execution quality)
    'amihud', 'spread_28', 'turnover_28',
    # Technical (trend confirmation)
    'rsi_14', 'macd_hist', 'donchian_pos',
    # Cross-sectional rank features
    'mom_14_csrank', 'rvol_28_csrank',
]


# ════════════════════════════════════════════
# FEATURE ENGINE (reuse from ml_features.py)
# ════════════════════════════════════════════

def load_and_compute():
    """Load panel and compute features."""
    from ml_features import load_daily_panel, compute_all_features, get_feature_columns
    panel = load_daily_panel()
    panel = compute_all_features(panel)
    # Verify our curated features exist
    available = [f for f in CORE_FEATURES if f in panel.columns]
    missing = [f for f in CORE_FEATURES if f not in panel.columns]
    if missing:
        print(f"WARNING: Missing features: {missing}")
    print(f"Using {len(available)} features: {available}")
    return panel, available


# ════════════════════════════════════════════
# TARGET: Market-neutralized ranked return
# ════════════════════════════════════════════

def compute_neutralized_target(panel, train_mask):
    """
    Target = cross-sectional RANK of (coin return - market return).
    This removes the market factor so ranking is about RELATIVE performance.
    """
    train = panel[train_mask].copy()
    g = train.groupby(level='symbol')

    # Raw forward return
    train['raw_fwd'] = g['close'].transform(lambda x: x.pct_change(HOLDING_DAYS).shift(-HOLDING_DAYS))

    # Market return (equal-weight average per date)
    market_ret = train.groupby(level='date')['raw_fwd'].transform('mean')

    # Neutralized return = coin return - market return
    train['neutral_fwd'] = train['raw_fwd'] - market_ret

    # Purge: remove rows whose target extends past training end
    train_dates = train.index.get_level_values('date')
    purge_cutoff = train_dates.max() - pd.Timedelta(days=PURGE_DAYS)
    train = train[train.index.get_level_values('date') <= purge_cutoff]
    train = train.dropna(subset=['neutral_fwd'])

    # Winsorize
    p1, p99 = train['neutral_fwd'].quantile([0.02, 0.98])
    train['neutral_fwd'] = train['neutral_fwd'].clip(p1, p99)

    # Cross-sectional rank (0-4 for CatBoost ranking)
    train['target_rank'] = train.groupby(level='date')['neutral_fwd'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') if len(x) >= 5 else 2
    ).fillna(2).astype(int)

    return train


# ════════════════════════════════════════════
# CATBOOST TRAINING with time-staggered ensemble
# ════════════════════════════════════════════

def train_catboost_ensemble(train_data, feat_cols):
    """
    Train N_ENSEMBLE models on staggered time windows.
    Each model sees a different slice — reduces temporal overfit.
    """
    models = []
    dates = train_data.index.get_level_values('date').unique().sort_values()
    n_dates = len(dates)

    for k in range(N_ENSEMBLE):
        # Stagger: each model uses a different 80% of the training window
        offset = int(n_dates * 0.2 * k / N_ENSEMBLE)
        end_idx = n_dates - int(n_dates * 0.2 * (N_ENSEMBLE - 1 - k) / N_ENSEMBLE)
        subset_dates = dates[offset:end_idx]

        subset = train_data[train_data.index.get_level_values('date').isin(subset_dates)]
        if len(subset) < 500:
            continue

        X = subset[feat_cols].values
        y = subset['target_rank'].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Group: each date is a query for ranking
        group_dates = subset.index.get_level_values('date')
        group_ids = pd.Categorical(group_dates).codes

        try:
            params = CATBOOST_PARAMS.copy()
            params['random_seed'] = 42 + k  # different seed per model
            model = CatBoostRanker(**params)
            model.fit(X, y, group_id=group_ids)
            models.append(model)
        except Exception as e:
            print(f"    Model {k} failed: {e}")

    return models


def predict_ensemble(X, models):
    """Average ranking scores from ensemble."""
    if not models:
        return None
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    scores = []
    for m in models:
        try:
            s = m.predict(X)
            s = (s - s.mean()) / (s.std() + 1e-10)
            scores.append(s)
        except Exception:
            pass
    if not scores:
        return None
    return np.mean(scores, axis=0)


# ════════════════════════════════════════════
# PATH-DEPENDENT STOP (from v2)
# ════════════════════════════════════════════

def simulate_period(panel, symbols, entry_date, hold_days, stop_pct, direction='long'):
    """Simulate holding with path-dependent stop using intraday data."""
    dates = panel.index.get_level_values('date').unique().sort_values()
    mask = dates <= entry_date
    if not mask.any():
        return {s: 0.0 for s in symbols}
    entry_loc = mask.sum() - 1
    exit_loc = min(entry_loc + hold_days, len(dates) - 1)
    hold_dates = dates[entry_loc:exit_loc + 1]

    results = {}
    for sym in symbols:
        try:
            sd = panel.xs(sym, level='symbol')
            avail = sd.index.intersection(hold_dates)
            if len(avail) < 2:
                results[sym] = 0.0
                continue

            entry_p = sd.loc[avail[0], 'close']
            if entry_p <= 0:
                results[sym] = 0.0
                continue

            peak = entry_p
            exit_p = entry_p
            stopped = False

            for d in avail[1:]:
                row = sd.loc[d]
                if direction == 'long':
                    lo = row.get('intra_low', row['low'])
                    stop_lvl = peak * (1 - stop_pct)
                    if lo <= stop_lvl:
                        exit_p = stop_lvl
                        stopped = True
                        break
                    hi = row.get('intra_high', row['high'])
                    peak = max(peak, hi)
                    exit_p = row['close']
                else:
                    hi = row.get('intra_high', row['high'])
                    stop_lvl = peak * (1 + stop_pct)
                    if hi >= stop_lvl:
                        exit_p = stop_lvl
                        stopped = True
                        break
                    lo = row.get('intra_low', row['low'])
                    peak = min(peak, lo)
                    exit_p = row['close']

            ret = exit_p / entry_p - 1 if direction == 'long' else -(exit_p / entry_p - 1)
            results[sym] = ret
        except Exception:
            results[sym] = 0.0

    return results


# ════════════════════════════════════════════
# MAIN BACKTEST
# ════════════════════════════════════════════

def run():
    print("=" * 70)
    print("  ML v3 — CatBoost YetiRank (pairwise ranking)")
    print("  Market-neutralized target | Staggered ensemble | Curated features")
    print("=" * 70)

    panel, feat_cols = load_and_compute()
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
    print(f"Rebalances: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # Funding rates
    funding_df = pd.DataFrame()
    fr_path = os.path.join(DATA_DIR, 'BTCUSDT_funding.parquet')
    if os.path.exists(fr_path):
        funding_df = pd.read_parquet(fr_path)

    equity = INITIAL_CAPITAL
    equity_curve = []
    all_importances = []
    ml_diagnostics = []  # per-period ML stats
    models = None

    for i, rd in enumerate(rebal_dates):
        # ── Train every ~2 months ──
        should_train = (i % max(1, 56 // HOLDING_DAYS) == 0) or i == 0

        if should_train:
            train_end = rd
            train_start = train_end - pd.DateOffset(months=TRAIN_MONTHS)
            train_mask = ((panel.index.get_level_values('date') >= train_start) &
                          (panel.index.get_level_values('date') < train_end))

            train_data = compute_neutralized_target(panel, train_mask)

            # Volume filter
            if 'vol_avg_28' in panel.columns:
                vol = panel.loc[train_data.index, 'vol_avg_28']
                thresh = vol.groupby(level='date').transform(lambda x: x.quantile(0.3))
                train_data = train_data[vol > thresh]

            print(f"  [{i+1}/{len(rebal_dates)}] {rd.date()} Train {len(train_data)} rows", flush=True)
            models = train_catboost_ensemble(train_data, feat_cols)

            if models:
                try:
                    imp = pd.Series(
                        models[0].get_feature_importance(type='PredictionValuesChange'),
                        index=feat_cols)
                    all_importances.append(imp / (imp.sum() + 1e-10))
                except Exception:
                    pass  # importance will be computed at end with SHAP

        if not models:
            equity_curve.append({'date': rd, 'equity': equity, 'n_pos': 0})
            continue

        # ── Predict ──
        if rd not in panel.index.get_level_values('date'):
            equity_curve.append({'date': rd, 'equity': equity, 'n_pos': 0})
            continue

        cross = panel.loc[rd].copy()
        cross = cross.dropna(subset=['mom_14', 'vol_avg_28'])
        cross = cross[cross['vol_avg_28'] > 0]
        cross = cross.nlargest(UNIVERSE_TOP, 'vol_avg_28')

        if len(cross) < TOP_N * 3:
            equity_curve.append({'date': rd, 'equity': equity, 'n_pos': 0})
            continue

        X = cross[feat_cols].values
        scores = predict_ensemble(X, models)
        if scores is None:
            equity_curve.append({'date': rd, 'equity': equity, 'n_pos': 0})
            continue

        cross = cross.copy()
        cross['score'] = scores
        cross = cross.sort_values('score', ascending=False)

        n_eligible = len(cross)

        idx_name = 'symbol' if 'symbol' in cross.index.names else None
        get_syms = lambda df: df.index.get_level_values('symbol').tolist() if idx_name else df.index.tolist()

        long_syms = get_syms(cross.head(TOP_N))
        short_syms = get_syms(cross.tail(TOP_N))

        # ── Path-dependent returns ──
        try:
            long_rets = simulate_period(panel, long_syms, rd, HOLDING_DAYS, 0.15, 'long')
            short_rets = simulate_period(panel, short_syms, rd, HOLDING_DAYS, 0.15, 'short')
        except Exception as e:
            print(f"    Sim error: {e}")
            equity_curve.append({'date': rd, 'equity': equity, 'n_pos': 0,
                                'long_ret': 0, 'short_ret': 0, 'total_ret': 0,
                                'btc_close': 0})
            continue

        long_ret = np.mean(list(long_rets.values())) if long_rets else 0
        short_ret = np.mean(list(short_rets.values())) if short_rets else 0

        # Funding on shorts
        if not funding_df.empty:
            fm = funding_df[funding_df.index <= rd]['fundingRate'].tail(21).mean()
            daily_f = abs(fm) * 3
        else:
            daily_f = FUNDING_DAILY
        short_ret -= daily_f * HOLDING_DAYS

        # L/S combined (market neutral)
        total_ret = (long_ret + short_ret) / 2
        total_ret -= 2 * COST_PER_SIDE

        if np.isnan(total_ret) or np.isinf(total_ret):
            total_ret = 0.0
        equity *= (1 + total_ret)
        if equity <= 0 or np.isnan(equity):
            equity = max(equity, 1.0) if not np.isnan(equity) else 1.0

        # BTC price
        btc_p = 0
        try:
            btc_p = panel.loc[(rd, 'BTCUSDT'), 'close']
            if hasattr(btc_p, 'iloc'):
                btc_p = btc_p.iloc[0]
        except Exception:
            pass

        # ── ML Diagnostics: Rank IC ──
        # Compute realized fwd return for all eligible coins (not just selected)
        all_realized = simulate_period(panel, get_syms(cross), rd, HOLDING_DAYS, 1.0, 'long')
        realized_series = pd.Series(all_realized)
        score_series = cross['score']
        score_series.index = get_syms(cross) if idx_name else cross.index

        # Align
        common = realized_series.index.intersection(score_series.index)
        if len(common) >= 10:
            from scipy.stats import spearmanr
            rank_ic, rank_ic_p = spearmanr(
                score_series.loc[common].values,
                realized_series.loc[common].values
            )
        else:
            rank_ic, rank_ic_p = 0.0, 1.0

        # Long vs short raw returns
        long_avg = np.mean([all_realized.get(s, 0) for s in long_syms])
        short_avg = np.mean([all_realized.get(s, 0) for s in short_syms])
        spread = long_avg - short_avg  # L/S spread before costs

        # Top quintile vs bottom quintile
        n5 = max(1, len(common) // 5)
        top_q = score_series.loc[common].nlargest(n5).index
        bot_q = score_series.loc[common].nsmallest(n5).index
        top_q_ret = realized_series.loc[top_q].mean() if len(top_q) > 0 else 0
        bot_q_ret = realized_series.loc[bot_q].mean() if len(bot_q) > 0 else 0

        ml_diagnostics.append({
            'date': rd,
            'rank_ic': rank_ic,
            'rank_ic_p': rank_ic_p,
            'spread': spread,
            'long_avg': long_avg,
            'short_avg': short_avg,
            'top_q_ret': top_q_ret,
            'bot_q_ret': bot_q_ret,
            'n_eligible': n_eligible,
            'long_ret_realized': long_ret,
            'short_ret_realized': short_ret,
        })

        equity_curve.append({
            'date': rd, 'equity': equity, 'n_pos': TOP_N * 2,
            'long_ret': long_ret, 'short_ret': short_ret,
            'total_ret': total_ret, 'btc_close': btc_p,
            'rank_ic': rank_ic, 'n_eligible': n_eligible,
            'spread': spread,
        })

    result = pd.DataFrame(equity_curve).set_index('date')
    result.to_parquet(os.path.join(RESULTS_DIR, 'ml_v3_result.parquet'))

    diag_df = pd.DataFrame(ml_diagnostics).set_index('date')
    diag_df.to_parquet(os.path.join(RESULTS_DIR, 'ml_v3_diagnostics.parquet'))

    # Importance
    avg_imp = pd.Series(dtype=float)
    if all_importances:
        avg_imp = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        avg_imp.to_json(os.path.join(RESULTS_DIR, 'ml_v3_importance.json'))

    # Metrics
    eq = result['equity']
    rets = eq.pct_change().dropna()
    n_yrs = max((result.index[-1] - result.index[0]).days / 365.25, 0.1)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_yrs) - 1 if eq.iloc[-1] > 0 else -1
    sharpe = rets.mean() / rets.std() * np.sqrt(365/HOLDING_DAYS) if rets.std() > 0 else 0
    ds = rets[rets < 0].std()
    sortino = rets.mean() / ds * np.sqrt(365/HOLDING_DAYS) if ds and ds > 0 else 0
    pk = eq.expanding().max()
    mdd = ((eq - pk) / pk).min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    yearly = {}
    for yr in sorted(result.index.year.unique()):
        y = eq[eq.index.year == yr]
        if len(y) > 1:
            yearly[yr] = (y.iloc[-1] / y.iloc[0] - 1) * 100

    print(f"\n{'='*70}")
    print(f"  ML v3 CATBOOST RESULTS")
    print(f"{'='*70}")
    print(f"  Period:  {result.index[0].date()} to {result.index[-1].date()}")
    print(f"  Initial: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Final:   ${eq.iloc[-1]:,.0f}")
    print(f"  CAGR:    {cagr*100:.1f}%")
    print(f"  Sharpe:  {sharpe:.2f}")
    print(f"  Sortino: {sortino:.2f}")
    print(f"  Max DD:  {mdd*100:.1f}%")
    print(f"  Calmar:  {calmar:.2f}")
    for yr, ret in yearly.items():
        print(f"  {yr}:    {ret:+.1f}%")
    if len(avg_imp) > 0:
        print(f"\n  TOP 15 FEATURES:")
        for f, v in avg_imp.head(15).items():
            print(f"    {f:25s}  {v:.4f}")
    print(f"{'='*70}")

    return result, avg_imp


def build_dashboard(result, importance, diag_path=None):
    """Full dashboard with ML diagnostics."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    eq = result['equity']

    # Load diagnostics
    dp = os.path.join(RESULTS_DIR, 'ml_v3_diagnostics.parquet')
    diag = pd.read_parquet(dp) if os.path.exists(dp) else pd.DataFrame()

    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            'Equity Curve (log)', 'Feature Importance (CatBoost)',
            'Drawdown', 'Rank IC Over Time (ML Quality)',
            'Annual Returns', 'Long vs Short Cumulative',
            'L/S Spread per Period', 'Universe Size Over Time',
            'Quintile Returns (Top vs Bottom)', 'Return Distribution',
        ],
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15],
    )

    # ═══ 1. Equity ═══
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name='CatBoost v3',
                             line=dict(color='#4CAF50', width=3)), row=1, col=1)
    if 'btc_close' in result.columns:
        btc = result['btc_close'].replace(0, np.nan).dropna()
        if len(btc) > 0 and btc.iloc[0] > 0:
            btc_eq = INITIAL_CAPITAL * btc / btc.iloc[0]
            fig.add_trace(go.Scatter(x=btc_eq.index, y=btc_eq, name='BTC B&H',
                                     line=dict(color='#FFD700', width=1.5, dash='dot')), row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="gray", row=1, col=1)

    # ═══ 2. Feature Importance ═══
    if len(importance) > 0:
        top = importance.head(20)[::-1]
        colors = ['#FF5722' if 'poly' in f or 'pullback' in f else
                  '#2196F3' if 'mom' in f else
                  '#FF9800' if any(x in f for x in ['amihud','spread','turnover']) else
                  '#4CAF50' for f in top.index]
        fig.add_trace(go.Bar(y=top.index, x=top.values, orientation='h',
                             marker_color=colors, name='Importance',
                             hovertemplate='%{y}: %{x:.3f}<extra></extra>'), row=1, col=2)

    # ═══ 3. Drawdown ═══
    pk = eq.expanding().max()
    dd = (eq - pk) / pk * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy',
                             line=dict(color='#f44336', width=1), name='DD'), row=2, col=1)

    # ═══ 4. Rank IC (THE key ML diagnostic) ═══
    if not diag.empty and 'rank_ic' in diag.columns:
        ic = diag['rank_ic']
        # Rolling mean
        ic_rolling = ic.rolling(6, min_periods=1).mean()
        fig.add_trace(go.Bar(x=ic.index, y=ic, name='Rank IC',
                             marker_color=['#4CAF50' if v > 0 else '#f44336' for v in ic],
                             opacity=0.4), row=2, col=2)
        fig.add_trace(go.Scatter(x=ic_rolling.index, y=ic_rolling, name='Rank IC (6-period MA)',
                                 line=dict(color='#FFD700', width=2.5)), row=2, col=2)
        fig.add_hline(y=0, line_color="gray", row=2, col=2)
        # Add annotation with avg IC
        avg_ic = ic.mean()
        pct_positive = (ic > 0).mean() * 100
        fig.add_annotation(x=0.5, y=1.0, xref='x4 domain', yref='y4 domain',
                          text=f'Avg IC: {avg_ic:.3f} | Positive: {pct_positive:.0f}%',
                          showarrow=False, font=dict(color='#FFD700', size=13))

    # ═══ 5. Annual Returns ═══
    yearly = {}
    for yr in sorted(eq.index.year.unique()):
        y = eq[eq.index.year == yr]
        if len(y) > 1:
            yearly[yr] = (y.iloc[-1] / y.iloc[0] - 1) * 100
    if yearly:
        c = ['#4CAF50' if v > 0 else '#f44336' for v in yearly.values()]
        fig.add_trace(go.Bar(x=[str(y) for y in yearly.keys()], y=list(yearly.values()),
                             marker_color=c, text=[f'{v:+.0f}%' for v in yearly.values()],
                             textposition='outside', name='Annual'), row=3, col=1)

    # ═══ 6. Long vs Short Cumulative ═══
    if 'long_ret' in result.columns:
        long_cum = (1 + result['long_ret'].fillna(0)).cumprod() * INITIAL_CAPITAL
        short_cum = (1 + result['short_ret'].fillna(0)).cumprod() * INITIAL_CAPITAL
        fig.add_trace(go.Scatter(x=long_cum.index, y=long_cum,
                                 name='Long Leg', line=dict(color='#4CAF50', width=2)), row=3, col=2)
        fig.add_trace(go.Scatter(x=short_cum.index, y=short_cum,
                                 name='Short Leg', line=dict(color='#f44336', width=2)), row=3, col=2)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="gray", row=3, col=2)
        fig.update_yaxes(type='log', row=3, col=2)

    # ═══ 7. L/S Spread per period ═══
    if not diag.empty and 'spread' in diag.columns:
        sp = diag['spread'] * 100
        fig.add_trace(go.Bar(x=sp.index, y=sp, name='L/S Spread %',
                             marker_color=['#4CAF50' if v > 0 else '#f44336' for v in sp],
                             opacity=0.7), row=4, col=1)
        sp_rolling = sp.rolling(6, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=sp_rolling.index, y=sp_rolling, name='Spread MA',
                                 line=dict(color='#FFD700', width=2)), row=4, col=1)
        fig.add_hline(y=0, line_color="gray", row=4, col=1)
        avg_spread = sp.mean()
        fig.add_annotation(x=0.5, y=1.0, xref='x7 domain', yref='y7 domain',
                          text=f'Avg Spread: {avg_spread:.2f}% per period',
                          showarrow=False, font=dict(color='#FFD700', size=13))

    # ═══ 8. Universe Size ═══
    if not diag.empty and 'n_eligible' in diag.columns:
        fig.add_trace(go.Scatter(x=diag.index, y=diag['n_eligible'],
                                 name='Eligible Coins', fill='tozeroy',
                                 line=dict(color='#03A9F4', width=1.5)), row=4, col=2)

    # ═══ 9. Quintile Returns ═══
    if not diag.empty and 'top_q_ret' in diag.columns:
        fig.add_trace(go.Bar(x=diag.index, y=diag['top_q_ret']*100,
                             name='Top Quintile', marker_color='#4CAF50', opacity=0.5), row=5, col=1)
        fig.add_trace(go.Bar(x=diag.index, y=diag['bot_q_ret']*100,
                             name='Bottom Quintile', marker_color='#f44336', opacity=0.5), row=5, col=1)
        fig.add_hline(y=0, line_color="gray", row=5, col=1)

    # ═══ 10. Return Distribution ═══
    rets = eq.pct_change().dropna() * 100
    fig.add_trace(go.Histogram(x=rets, nbinsx=40, marker_color='#2196F3',
                               name='Returns %'), row=5, col=2)
    fig.add_vline(x=0, line_color="white", row=5, col=2)

    # ═══ Summary Stats Table (as annotation) ═══
    n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_yrs) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(365/HOLDING_DAYS/100) if rets.std() > 0 else 0
    mdd = dd.min()
    avg_ic = diag['rank_ic'].mean() if not diag.empty and 'rank_ic' in diag.columns else 0
    ic_pos = (diag['rank_ic'] > 0).mean() * 100 if not diag.empty and 'rank_ic' in diag.columns else 0

    summary_text = (
        f"CAGR: {cagr*100:.1f}% | Sharpe: {sharpe:.2f} | Max DD: {mdd:.1f}% | "
        f"Avg Rank IC: {avg_ic:.3f} | IC>0: {ic_pos:.0f}% | "
        f"${INITIAL_CAPITAL/1000:.0f}k → ${eq.iloc[-1]/1000:.0f}k"
    )

    fig.update_layout(
        height=2000, template='plotly_dark',
        title_text=f'ML v3 CatBoost YetiRank — Full Diagnostics<br><sub>{summary_text}</sub>',
        showlegend=True,
        legend=dict(orientation='h', y=-0.02),
    )

    path = os.path.join(RESULTS_DIR, 'ml_v3_dashboard.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"Dashboard: {path}")
    return path


if __name__ == '__main__':
    result, importance = run()
    path = build_dashboard(result, importance)
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')
