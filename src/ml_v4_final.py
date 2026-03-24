"""
ML v4 FINAL — CatBoost Binary Loser Classifier + Dynamic N + Barbell Architecture
Implements the committee's approved plan:
  Phase 1: A1 (binary classifier for losers) + A3 (dynamic N shorts)
  Phase 2: C3 Barbell (short alpha + stablecoin yield)

All audit fixes from v2 preserved:
  - P&L from actual prices (no fwd_14)
  - Purge 16 days from training
  - Path-dependent trailing stop
  - Realistic costs (0.2%/side)
  - Real funding rates
  - No look-ahead bias
"""
import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
TRAIN_MONTHS = 18
PURGE_DAYS = 18
HOLDING_DAYS = 14
MIN_N_SHORTS = 3
MAX_N_SHORTS = 8
DEFAULT_N_SHORTS = 5
LOSER_THRESHOLD = 0.10     # bottom 10% = loser label
CONFIDENCE_GATE = 0.55     # minimum classifier probability to short
UNIVERSE_TOP = 50          # top 50 by volume
STOP_PCT = 0.15
COST_PER_SIDE = 0.002
FUNDING_DAILY_DEFAULT = 0.001
INITIAL_CAPITAL = 100_000
STABLECOIN_YIELD_ANNUAL = 0.05  # 5% for the barbell

# CatBoost — conservative, anti-overfit
CLASSIFIER_PARAMS = {
    'iterations': 250,
    'depth': 4,
    'learning_rate': 0.05,
    'l2_leaf_reg': 7.0,
    'random_strength': 2.0,
    'bagging_temperature': 1.0,
    'boosting_type': 'Ordered',
    'auto_class_weights': 'Balanced',  # handle class imbalance (10% losers)
    'eval_metric': 'AUC',
    'verbose': 0,
    'random_seed': 42,
    'task_type': 'CPU',
}

RANKER_PARAMS = {
    'loss_function': 'YetiRank',
    'iterations': 200,
    'depth': 4,
    'learning_rate': 0.05,
    'l2_leaf_reg': 5.0,
    'random_strength': 2.0,
    'bagging_temperature': 1.0,
    'verbose': 0,
    'random_seed': 42,
    'task_type': 'CPU',
}

# Curated features — 20 features only
FEATURES = [
    'mom_14', 'mom_28', 'mom_56', 'mom_14_skip1',
    'poly_slope_28', 'poly_curve_28', 'poly_slope_56', 'poly_curve_56',
    'rvol_28', 'vol_ratio', 'max_ret_28', 'min_ret_28',
    'amihud', 'spread_28', 'turnover_28',
    'rsi_14', 'macd_hist', 'donchian_pos',
    'mom_14_csrank', 'rvol_28_csrank',
]


# ═══════════════════════════════════════
# DATA LOADING + FEATURES (reuse from ml_features.py)
# ═══════════════════════════════════════

def load_and_compute():
    from ml_features import load_daily_panel, compute_all_features
    panel = load_daily_panel()
    panel = compute_all_features(panel)
    available = [f for f in FEATURES if f in panel.columns]
    print(f"Using {len(available)} features")
    return panel, available


# ═══════════════════════════════════════
# TRAINING: Binary Classifier + Ranker
# ═══════════════════════════════════════

def build_training_data(panel, train_mask):
    """Build training set with binary loser labels + purging."""
    train = panel[train_mask].copy()
    g = train.groupby(level='symbol')

    # Forward return
    train['fwd'] = g['close'].transform(lambda x: x.pct_change(HOLDING_DAYS).shift(-HOLDING_DAYS))

    # Purge
    max_date = train.index.get_level_values('date').max()
    cutoff = max_date - pd.Timedelta(days=PURGE_DAYS)
    train = train[train.index.get_level_values('date') <= cutoff]
    train = train.dropna(subset=['fwd'])

    # Market-neutralized return
    mkt_ret = train.groupby(level='date')['fwd'].transform('mean')
    train['neutral_fwd'] = train['fwd'] - mkt_ret

    # Binary label: bottom LOSER_THRESHOLD = 1, rest = 0
    train['is_loser'] = train.groupby(level='date')['neutral_fwd'].transform(
        lambda x: (x <= x.quantile(LOSER_THRESHOLD)).astype(int)
    )

    # Rank label for ranker (quintiles 0-4)
    train['rank_label'] = train.groupby(level='date')['neutral_fwd'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') if len(x) >= 5 else 2
    ).fillna(2).astype(int)

    # Winsorize fwd
    p2, p98 = train['neutral_fwd'].quantile([0.02, 0.98])
    train['neutral_fwd'] = train['neutral_fwd'].clip(p2, p98)

    # Volume filter
    if 'vol_avg_28' in panel.columns:
        vol = panel.loc[train.index, 'vol_avg_28']
        thresh = vol.groupby(level='date').transform(lambda x: x.quantile(0.25))
        train = train[vol > thresh]

    return train


def train_models(train_data, feat_cols):
    """Train binary classifier (A1) + ranker (backup)."""
    X = train_data[feat_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    if len(X) < 300:
        return None, None

    # ── A1: Binary Classifier (is this coin a loser?) ──
    y_binary = train_data['is_loser'].values
    classifier = None
    try:
        classifier = CatBoostClassifier(**CLASSIFIER_PARAMS)
        classifier.fit(X, y_binary)
    except Exception as e:
        print(f"    Classifier failed: {e}")

    # ── Ranker (backup/ensemble) ──
    y_rank = train_data['rank_label'].values
    dates = train_data.index.get_level_values('date')
    group_ids = pd.Categorical(dates).codes

    ranker = None
    try:
        ranker = CatBoostRanker(**RANKER_PARAMS)
        ranker.fit(X, y_rank, group_id=group_ids)
    except Exception as e:
        print(f"    Ranker failed: {e}")

    return classifier, ranker


# ═══════════════════════════════════════
# PREDICTION: Ensemble (classifier veto + ranker rank)
# ═══════════════════════════════════════

def predict_shorts(cross, feat_cols, classifier, ranker):
    """
    A1 + A3: Use classifier to identify losers, ranker to rank them.
    Dynamic N: only short coins where classifier is confident.
    """
    X = cross[feat_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    scores = np.zeros(len(X))
    loser_probs = np.full(len(X), 0.5)

    # Classifier: probability of being a loser
    if classifier is not None:
        try:
            probs = classifier.predict_proba(X)[:, 1]  # P(loser)
            loser_probs = probs
        except Exception:
            pass

    # Ranker: rank score (lower = worse coin)
    if ranker is not None:
        try:
            rank_scores = ranker.predict(X)
            # Normalize to [0,1]
            rank_norm = (rank_scores - rank_scores.min()) / (rank_scores.max() - rank_scores.min() + 1e-10)
            scores = rank_norm
        except Exception:
            pass

    cross = cross.copy()
    cross['loser_prob'] = loser_probs
    cross['rank_score'] = scores

    # ── A1: Filter by classifier confidence ──
    candidates = cross[cross['loser_prob'] >= CONFIDENCE_GATE].copy()

    if len(candidates) == 0:
        # Fallback: use ranker bottom N
        candidates = cross.nsmallest(DEFAULT_N_SHORTS, 'rank_score')

    # ── A3: Dynamic N based on how many confident losers ──
    n_confident = len(candidates)
    n_shorts = max(MIN_N_SHORTS, min(n_confident, MAX_N_SHORTS))

    # Sort by loser probability (highest first = most confident loser)
    candidates = candidates.sort_values('loser_prob', ascending=False)
    selected = candidates.head(n_shorts)

    return selected, n_shorts


# ═══════════════════════════════════════
# PATH-DEPENDENT STOP (from v2/v3)
# ═══════════════════════════════════════

def simulate_period(panel, symbols, entry_date, hold_days, stop_pct, direction='short'):
    """Path-dependent trailing stop using intraday high/low."""
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

            for d in avail[1:]:
                row = sd.loc[d]
                if direction == 'short':
                    hi = row.get('intra_high', row['high'])
                    stop_lvl = peak * (1 + stop_pct)
                    if hi >= stop_lvl:
                        exit_p = stop_lvl
                        break
                    lo = row.get('intra_low', row['low'])
                    peak = min(peak, lo)
                    exit_p = row['close']
                else:  # long
                    lo = row.get('intra_low', row['low'])
                    stop_lvl = peak * (1 - stop_pct)
                    if lo <= stop_lvl:
                        exit_p = stop_lvl
                        break
                    hi = row.get('intra_high', row['high'])
                    peak = max(peak, hi)
                    exit_p = row['close']

            if direction == 'short':
                ret = -(exit_p / entry_p - 1)
            else:
                ret = exit_p / entry_p - 1
            results[sym] = ret
        except Exception:
            results[sym] = 0.0
    return results


# ═══════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════

def run():
    print("=" * 70)
    print("  ML v4 FINAL — Binary Loser Classifier + Dynamic N + Barbell")
    print("  Committee-approved architecture (C3 + A1 + A3)")
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

    # ── Barbell allocation ──
    SHORT_ALLOC = 0.40   # 40% to short strategy
    STABLE_ALLOC = 0.60  # 60% to stablecoin yield

    equity = INITIAL_CAPITAL
    equity_curve = []
    ml_diagnostics = []
    all_importances = []
    classifier = ranker = None

    for i, rd in enumerate(rebal_dates):
        # ── Train every ~2 months ──
        should_train = (i % max(1, 56 // HOLDING_DAYS) == 0) or i == 0

        if should_train:
            train_end = rd
            train_start = train_end - pd.DateOffset(months=TRAIN_MONTHS)
            train_mask = ((panel.index.get_level_values('date') >= train_start) &
                          (panel.index.get_level_values('date') < train_end))

            train_data = build_training_data(panel, train_mask)
            n_losers = train_data['is_loser'].sum()
            n_total = len(train_data)

            print(f"  [{i+1}/{len(rebal_dates)}] {rd.date()} Train {n_total} rows "
                  f"({n_losers} losers, {n_losers/max(n_total,1)*100:.0f}%)", flush=True)

            classifier, ranker = train_models(train_data, feat_cols)

            if classifier is not None:
                try:
                    imp = pd.Series(
                        classifier.get_feature_importance(type='PredictionValuesChange'),
                        index=feat_cols)
                    all_importances.append(imp / (imp.sum() + 1e-10))
                except Exception:
                    pass

        if classifier is None and ranker is None:
            # No model yet — park in stables
            stable_yield = equity * STABLECOIN_YIELD_ANNUAL / 365 * HOLDING_DAYS
            equity += stable_yield
            equity_curve.append({'date': rd, 'equity': equity, 'n_shorts': 0,
                                'short_ret': 0, 'stable_yield': stable_yield,
                                'total_ret': stable_yield / equity, 'btc_close': 0,
                                'n_eligible': 0, 'rank_ic': 0, 'avg_loser_prob': 0})
            continue

        # ── Get cross-section ──
        if rd not in panel.index.get_level_values('date'):
            equity_curve.append({'date': rd, 'equity': equity, 'n_shorts': 0,
                                'short_ret': 0, 'stable_yield': 0, 'total_ret': 0,
                                'btc_close': 0, 'n_eligible': 0, 'rank_ic': 0,
                                'avg_loser_prob': 0})
            continue

        cross = panel.loc[rd].copy()
        cross = cross.dropna(subset=['mom_14', 'vol_avg_28'])
        cross = cross[cross['vol_avg_28'] > 0]
        cross = cross.nlargest(UNIVERSE_TOP, 'vol_avg_28')

        if len(cross) < 10:
            equity_curve.append({'date': rd, 'equity': equity, 'n_shorts': 0,
                                'short_ret': 0, 'stable_yield': 0, 'total_ret': 0,
                                'btc_close': 0, 'n_eligible': len(cross), 'rank_ic': 0,
                                'avg_loser_prob': 0})
            continue

        n_eligible = len(cross)

        # ── Predict: A1 classifier + A3 dynamic N ──
        selected, n_shorts = predict_shorts(cross, feat_cols, classifier, ranker)
        short_syms = selected.index.tolist()
        avg_loser_prob = selected['loser_prob'].mean()

        # ── Path-dependent short returns ──
        try:
            short_rets = simulate_period(panel, short_syms, rd, HOLDING_DAYS, STOP_PCT, 'short')
        except Exception:
            short_rets = {s: 0.0 for s in short_syms}

        short_ret = np.mean(list(short_rets.values())) if short_rets else 0

        # ── Funding cost (real) ──
        if not funding_df.empty:
            fm = funding_df[funding_df.index <= rd]['fundingRate'].tail(21).mean()
            daily_f = abs(fm) * 3
        else:
            daily_f = FUNDING_DAILY_DEFAULT
        short_ret -= daily_f * HOLDING_DAYS

        # ── Transaction costs ──
        short_ret -= 2 * COST_PER_SIDE

        if np.isnan(short_ret) or np.isinf(short_ret):
            short_ret = 0.0

        # ── Barbell: weighted return ──
        stable_yield = equity * STABLE_ALLOC * STABLECOIN_YIELD_ANNUAL / 365 * HOLDING_DAYS
        short_pnl = equity * SHORT_ALLOC * short_ret
        total_pnl = short_pnl + stable_yield

        equity += total_pnl
        if equity <= 0 or np.isnan(equity):
            equity = max(1.0, equity) if not np.isnan(equity) else 1.0

        # ── Rank IC diagnostic ──
        rank_ic = 0.0
        try:
            all_realized = simulate_period(panel, cross.index.tolist(), rd, HOLDING_DAYS, 1.0, 'short')
            realized = pd.Series(all_realized)
            probs = cross['loser_prob'] if 'loser_prob' in cross.columns else pd.Series(0, index=cross.index)
            common = realized.index.intersection(probs.index)
            if len(common) >= 10:
                ic, _ = spearmanr(probs.loc[common].values, -realized.loc[common].values)
                rank_ic = ic if not np.isnan(ic) else 0.0
        except Exception:
            pass

        # BTC price
        btc_p = 0
        try:
            btc_p = panel.loc[(rd, 'BTCUSDT'), 'close']
            if hasattr(btc_p, 'iloc'):
                btc_p = btc_p.iloc[0]
        except Exception:
            pass

        equity_curve.append({
            'date': rd, 'equity': equity, 'n_shorts': n_shorts,
            'short_ret': short_ret, 'stable_yield': stable_yield,
            'total_ret': total_pnl / max(equity - total_pnl, 1),
            'btc_close': btc_p, 'n_eligible': n_eligible,
            'rank_ic': rank_ic, 'avg_loser_prob': avg_loser_prob,
        })

        ml_diagnostics.append({
            'date': rd, 'rank_ic': rank_ic, 'n_shorts': n_shorts,
            'n_eligible': n_eligible, 'avg_loser_prob': avg_loser_prob,
            'short_ret': short_ret, 'stable_yield': stable_yield,
        })

    # ═══ RESULTS ═══
    result = pd.DataFrame(equity_curve).set_index('date')
    result.to_parquet(os.path.join(RESULTS_DIR, 'ml_v4_result.parquet'))

    diag = pd.DataFrame(ml_diagnostics).set_index('date')
    diag.to_parquet(os.path.join(RESULTS_DIR, 'ml_v4_diagnostics.parquet'))

    avg_imp = pd.Series(dtype=float)
    if all_importances:
        avg_imp = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        avg_imp.to_json(os.path.join(RESULTS_DIR, 'ml_v4_importance.json'))

    eq = result['equity']
    n_yrs = max((result.index[-1] - result.index[0]).days / 365.25, 0.1)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_yrs) - 1 if eq.iloc[-1] > 0 else -1
    rets = eq.pct_change().dropna()
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

    avg_ic = diag['rank_ic'].mean() if len(diag) > 0 else 0
    ic_pos = (diag['rank_ic'] > 0).mean() * 100 if len(diag) > 0 else 0
    avg_n = diag['n_shorts'].mean() if len(diag) > 0 else 0
    avg_prob = diag['avg_loser_prob'].mean() if len(diag) > 0 else 0

    print(f"\n{'='*70}")
    print(f"  ML v4 FINAL — BARBELL (Short Alpha + Stablecoin Yield)")
    print(f"{'='*70}")
    print(f"  Period:        {result.index[0].date()} to {result.index[-1].date()}")
    print(f"  Initial:       ${INITIAL_CAPITAL:,.0f}")
    print(f"  Final:         ${eq.iloc[-1]:,.0f}")
    print(f"  CAGR:          {cagr*100:.1f}%")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Sortino:       {sortino:.2f}")
    print(f"  Max DD:        {mdd*100:.1f}%")
    print(f"  Calmar:        {calmar:.2f}")
    print(f"  Allocation:    {SHORT_ALLOC*100:.0f}% short / {STABLE_ALLOC*100:.0f}% stables")
    for yr, ret in yearly.items():
        print(f"  {yr}:          {ret:+.1f}%")
    print(f"\n  ML DIAGNOSTICS:")
    print(f"  Avg Rank IC:   {avg_ic:.4f}")
    print(f"  IC > 0:        {ic_pos:.0f}%")
    print(f"  Avg N shorts:  {avg_n:.1f}")
    print(f"  Avg P(loser):  {avg_prob:.3f}")
    if len(avg_imp) > 0:
        print(f"\n  TOP 15 FEATURES (Classifier):")
        for f, v in avg_imp.head(15).items():
            print(f"    {f:25s}  {v:.4f}")
    print(f"{'='*70}")

    return result, avg_imp, diag


# ═══════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════

def build_dashboard(result, importance, diag):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    SHORT_ALLOC = 0.40
    STABLE_ALLOC = 0.60
    eq = result['equity']
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            'Equity: Barbell (Short Alpha + 5% Yield)', 'Classifier Feature Importance',
            'Drawdown', 'Rank IC (Classifier Quality)',
            'Annual Returns', 'Short Leg Cumulative',
            'Dynamic N (shorts per period)', 'Classifier Confidence',
            'Short Return Distribution', 'Yield vs Short PnL Decomposition',
        ],
        row_heights=[0.25, 0.2, 0.2, 0.15, 0.2],
    )

    # 1. Equity
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name='Barbell v4',
                             line=dict(color='#4CAF50', width=3)), row=1, col=1)
    if 'btc_close' in result.columns:
        btc = result['btc_close'].replace(0, np.nan).dropna()
        if len(btc) > 0 and btc.iloc[0] > 0:
            btc_eq = INITIAL_CAPITAL * btc / btc.iloc[0]
            fig.add_trace(go.Scatter(x=btc_eq.index, y=btc_eq, name='BTC B&H',
                                     line=dict(color='#FFD700', width=1.5, dash='dot')), row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="gray", row=1, col=1)

    # 2. Feature importance
    if len(importance) > 0:
        top = importance.head(20)[::-1]
        colors = ['#FF5722' if 'poly' in f else '#2196F3' if 'mom' in f else
                  '#FF9800' if any(x in f for x in ['amihud','spread','turn']) else '#4CAF50'
                  for f in top.index]
        fig.add_trace(go.Bar(y=top.index, x=top.values, orientation='h',
                             marker_color=colors, name='Classifier Imp.'), row=1, col=2)

    # 3. Drawdown
    pk = eq.expanding().max()
    dd = (eq - pk) / pk * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy',
                             line=dict(color='#f44336', width=1), name='DD'), row=2, col=1)
    fig.add_hline(y=-15, line_dash="dash", line_color="yellow",
                  annotation_text="Circuit Breaker -15%", row=2, col=1)

    # 4. Rank IC
    if len(diag) > 0 and 'rank_ic' in diag.columns:
        ic = diag['rank_ic']
        ic_ma = ic.rolling(6, min_periods=1).mean()
        fig.add_trace(go.Bar(x=ic.index, y=ic, name='IC',
                             marker_color=['#4CAF50' if v > 0 else '#f44336' for v in ic],
                             opacity=0.4), row=2, col=2)
        fig.add_trace(go.Scatter(x=ic_ma.index, y=ic_ma, name='IC MA(6)',
                                 line=dict(color='#FFD700', width=2.5)), row=2, col=2)
        fig.add_hline(y=0, line_color="gray", row=2, col=2)
        fig.add_hline(y=0.03, line_dash="dash", line_color="red",
                      annotation_text="Min IC threshold", row=2, col=2)

    # 5. Annual returns
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

    # 6. Short cumulative
    if 'short_ret' in result.columns:
        short_cum = (1 + result['short_ret'].fillna(0) * 0.40).cumprod() * INITIAL_CAPITAL
        fig.add_trace(go.Scatter(x=short_cum.index, y=short_cum, name='Short Leg (40%)',
                                 line=dict(color='#f44336', width=2)), row=3, col=2)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="gray", row=3, col=2)

    # 7. Dynamic N
    if len(diag) > 0:
        fig.add_trace(go.Scatter(x=diag.index, y=diag['n_shorts'], name='N shorts',
                                 mode='lines+markers', line=dict(color='#03A9F4')), row=4, col=1)
        fig.add_hline(y=MIN_N_SHORTS, line_dash="dash", line_color="gray", row=4, col=1)
        fig.add_hline(y=MAX_N_SHORTS, line_dash="dash", line_color="gray", row=4, col=1)

    # 8. Classifier confidence
    if len(diag) > 0:
        fig.add_trace(go.Scatter(x=diag.index, y=diag['avg_loser_prob'], name='Avg P(loser)',
                                 line=dict(color='#FF9800', width=2)), row=4, col=2)
        fig.add_hline(y=CONFIDENCE_GATE, line_dash="dash", line_color="red",
                      annotation_text=f"Gate={CONFIDENCE_GATE}", row=4, col=2)

    # 9. Short return distribution
    if 'short_ret' in result.columns:
        srets = result['short_ret'].dropna() * 100
        fig.add_trace(go.Histogram(x=srets, nbinsx=40, marker_color='#f44336',
                                   name='Short Ret %'), row=5, col=1)
        fig.add_vline(x=0, line_color="white", row=5, col=1)

    # 10. Yield vs Short PnL
    if 'short_ret' in result.columns and 'stable_yield' in result.columns:
        cum_short = (result['short_ret'].fillna(0) * 0.40 * INITIAL_CAPITAL).cumsum()
        cum_yield = result['stable_yield'].fillna(0).cumsum()
        fig.add_trace(go.Scatter(x=result.index, y=cum_short, name='Short PnL (cumul)',
                                 line=dict(color='#f44336')), row=5, col=2)
        fig.add_trace(go.Scatter(x=result.index, y=cum_yield, name='Yield PnL (cumul)',
                                 line=dict(color='#4CAF50')), row=5, col=2)

    # Summary in title
    n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_yrs) - 1
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(365/HOLDING_DAYS) if rets.std() > 0 else 0
    avg_ic = diag['rank_ic'].mean() if len(diag) > 0 else 0

    fig.update_layout(
        height=2200, template='plotly_dark',
        title_text=(f'ML v4 FINAL — Barbell: Short Alpha ({SHORT_ALLOC*100:.0f}%) + '
                    f'Stablecoin Yield ({STABLE_ALLOC*100:.0f}%)<br>'
                    f'<sub>CAGR: {cagr*100:.1f}% | Sharpe: {sharpe:.2f} | '
                    f'Max DD: {((eq-eq.expanding().max())/eq.expanding().max()).min()*100:.1f}% | '
                    f'Avg Rank IC: {avg_ic:.3f} | '
                    f'${INITIAL_CAPITAL/1000:.0f}k → ${eq.iloc[-1]/1000:.0f}k</sub>'),
        showlegend=True,
    )

    path = os.path.join(RESULTS_DIR, 'ml_v4_dashboard.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"Dashboard: {path}")
    return path


if __name__ == '__main__':
    result, importance, diag = run()
    path = build_dashboard(result, importance, diag)
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')
