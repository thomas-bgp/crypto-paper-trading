"""
ML v5 — SHORT-ONLY IMPROVED. Must beat baseline Sharpe 1.36 / +933%.
Changes from v3 baseline:
  - A1: Binary classifier as VETO GATE (filter bad shorts out)
  - A3: Dynamic N (3-8 based on classifier confidence)
  - Ranker still does the ranking — classifier only filters
  - 100% allocated to shorts (NO barbell dilution)
  - Confidence-weighted sizing (more $ on highest-conviction shorts)
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, CatBoostRanker
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══ CONFIG ═══
TRAIN_MONTHS = 18
PURGE_DAYS = 18
HOLDING_DAYS = 14
MIN_N = 3
MAX_N = 8
LOSER_THRESHOLD = 0.10
CONFIDENCE_GATE = 0.50
UNIVERSE_TOP = 50
STOP_PCT = 0.15
COST_PER_SIDE = 0.002
FUNDING_DAILY_DEFAULT = 0.001
INITIAL_CAPITAL = 100_000

CLASSIFIER_PARAMS = {
    'iterations': 250, 'depth': 4, 'learning_rate': 0.05,
    'l2_leaf_reg': 7.0, 'random_strength': 2.0, 'bagging_temperature': 1.0,
    'boosting_type': 'Ordered', 'auto_class_weights': 'Balanced',
    'eval_metric': 'AUC', 'verbose': 0, 'random_seed': 42, 'task_type': 'CPU',
}
RANKER_PARAMS = {
    'loss_function': 'YetiRank', 'iterations': 200, 'depth': 4,
    'learning_rate': 0.05, 'l2_leaf_reg': 5.0, 'random_strength': 2.0,
    'bagging_temperature': 1.0, 'verbose': 0, 'random_seed': 42, 'task_type': 'CPU',
}

FEATURES = [
    'mom_14', 'mom_28', 'mom_56', 'mom_14_skip1',
    'poly_slope_28', 'poly_curve_28', 'poly_slope_56', 'poly_curve_56',
    'rvol_28', 'vol_ratio', 'max_ret_28', 'min_ret_28',
    'amihud', 'spread_28', 'turnover_28',
    'rsi_14', 'macd_hist', 'donchian_pos',
    'mom_14_csrank', 'rvol_28_csrank',
]


def load_and_compute():
    from ml_features import load_daily_panel, compute_all_features
    panel = load_daily_panel()
    panel = compute_all_features(panel)
    avail = [f for f in FEATURES if f in panel.columns]
    print(f"Features: {len(avail)}")
    return panel, avail


def build_training(panel, train_mask):
    train = panel[train_mask].copy()
    g = train.groupby(level='symbol')
    train['fwd'] = g['close'].transform(lambda x: x.pct_change(HOLDING_DAYS).shift(-HOLDING_DAYS))
    cutoff = train.index.get_level_values('date').max() - pd.Timedelta(days=PURGE_DAYS)
    train = train[train.index.get_level_values('date') <= cutoff]
    train = train.dropna(subset=['fwd'])
    mkt = train.groupby(level='date')['fwd'].transform('mean')
    train['nfwd'] = train['fwd'] - mkt
    p2, p98 = train['nfwd'].quantile([0.02, 0.98])
    train['nfwd'] = train['nfwd'].clip(p2, p98)
    train['is_loser'] = train.groupby(level='date')['nfwd'].transform(
        lambda x: (x <= x.quantile(LOSER_THRESHOLD)).astype(int))
    train['rank_label'] = train.groupby(level='date')['nfwd'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') if len(x) >= 5 else 2
    ).fillna(2).astype(int)
    if 'vol_avg_28' in panel.columns:
        vol = panel.loc[train.index, 'vol_avg_28']
        thresh = vol.groupby(level='date').transform(lambda x: x.quantile(0.25))
        train = train[vol > thresh]
    return train


def train_models(train_data, feat_cols):
    X = np.nan_to_num(train_data[feat_cols].values, nan=0, posinf=0, neginf=0)
    if len(X) < 300:
        return None, None

    # Classifier (loser detector)
    clf = None
    try:
        clf = CatBoostClassifier(**CLASSIFIER_PARAMS)
        clf.fit(X, train_data['is_loser'].values)
    except Exception as e:
        print(f"    Clf fail: {e}")

    # Ranker (for ordering within the filtered set)
    rnk = None
    try:
        dates = train_data.index.get_level_values('date')
        gids = pd.Categorical(dates).codes
        rnk = CatBoostRanker(**RANKER_PARAMS)
        rnk.fit(X, train_data['rank_label'].values, group_id=gids)
    except Exception as e:
        print(f"    Rnk fail: {e}")

    return clf, rnk


def select_shorts(cross, feat_cols, clf, rnk):
    """
    Two-stage selection:
    1. Classifier FILTERS: only coins with P(loser) > CONFIDENCE_GATE pass
    2. Ranker RANKS the filtered set: short the worst-ranked among confirmed losers
    3. Dynamic N: number of shorts = number that pass the gate (clamped 3-8)
    4. Confidence-weighted sizing: higher P(loser) → bigger position
    """
    X = np.nan_to_num(cross[feat_cols].values, nan=0, posinf=0, neginf=0)

    # Stage 1: Classifier gate
    loser_probs = np.full(len(X), 0.5)
    if clf is not None:
        try:
            loser_probs = clf.predict_proba(X)[:, 1]
        except Exception:
            pass

    cross = cross.copy()
    cross['loser_prob'] = loser_probs

    # Only keep coins classifier thinks are losers
    candidates = cross[cross['loser_prob'] >= CONFIDENCE_GATE]

    if len(candidates) < MIN_N:
        # Fallback: take top MIN_N by loser probability
        candidates = cross.nlargest(MIN_N, 'loser_prob')

    # Stage 2: Ranker orders the filtered candidates
    rank_scores = np.zeros(len(candidates))
    if rnk is not None:
        try:
            Xc = np.nan_to_num(candidates[feat_cols].values, nan=0, posinf=0, neginf=0)
            rank_scores = rnk.predict(Xc)
        except Exception:
            pass

    candidates = candidates.copy()
    candidates['rank_score'] = rank_scores

    # Dynamic N: clamp to [MIN_N, MAX_N]
    n = max(MIN_N, min(len(candidates), MAX_N))

    # Take the n WORST by rank_score (lowest score = worst coin)
    selected = candidates.nsmallest(n, 'rank_score')

    # Confidence-weighted sizing: normalize P(loser) to weights
    probs = selected['loser_prob'].values
    weights = probs / probs.sum()  # higher prob → bigger weight

    return selected, weights, n


def simulate_period(panel, symbols, entry_date, hold_days, stop_pct):
    """Path-dependent short with trailing stop."""
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
                hi = row.get('intra_high', row['high'])
                stop_lvl = peak * (1 + stop_pct)
                if hi >= stop_lvl:
                    exit_p = stop_lvl
                    break
                lo = row.get('intra_low', row['low'])
                peak = min(peak, lo)
                exit_p = row['close']
            results[sym] = -(exit_p / entry_p - 1)  # short profit
        except Exception:
            results[sym] = 0.0
    return results


def run():
    print("=" * 70)
    print("  ML v5 — SHORT-ONLY IMPROVED")
    print("  Target: beat baseline Sharpe 1.36 / +933%")
    print("  Classifier FILTER + Ranker ORDER + Dynamic N + Confidence sizing")
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
    print(f"Rebalances: {len(rebal_dates)}")

    funding_df = pd.DataFrame()
    fr_path = os.path.join(DATA_DIR, 'BTCUSDT_funding.parquet')
    if os.path.exists(fr_path):
        funding_df = pd.read_parquet(fr_path)

    equity = INITIAL_CAPITAL
    equity_curve = []
    diagnostics = []
    all_imp = []
    clf = rnk = None

    for i, rd in enumerate(rebal_dates):
        should_train = (i % max(1, 56 // HOLDING_DAYS) == 0) or i == 0
        if should_train:
            te = rd
            ts = te - pd.DateOffset(months=TRAIN_MONTHS)
            tmask = ((panel.index.get_level_values('date') >= ts) &
                     (panel.index.get_level_values('date') < te))
            td = build_training(panel, tmask)
            print(f"  [{i+1}/{len(rebal_dates)}] {rd.date()} Train {len(td)} rows "
                  f"({td['is_loser'].sum()} losers)", flush=True)
            clf, rnk = train_models(td, feat_cols)
            if clf is not None:
                try:
                    imp = pd.Series(clf.get_feature_importance(type='PredictionValuesChange'),
                                    index=feat_cols)
                    all_imp.append(imp / (imp.sum() + 1e-10))
                except Exception:
                    pass

        if clf is None and rnk is None:
            equity_curve.append({'date': rd, 'equity': equity, 'n': 0, 'ret': 0,
                                'btc': 0, 'ic': 0, 'avg_prob': 0, 'n_elig': 0})
            continue

        if rd not in panel.index.get_level_values('date'):
            equity_curve.append({'date': rd, 'equity': equity, 'n': 0, 'ret': 0,
                                'btc': 0, 'ic': 0, 'avg_prob': 0, 'n_elig': 0})
            continue

        cross = panel.loc[rd].copy()
        cross = cross.dropna(subset=['mom_14', 'vol_avg_28'])
        cross = cross[cross['vol_avg_28'] > 0]
        cross = cross.nlargest(UNIVERSE_TOP, 'vol_avg_28')

        if len(cross) < 10:
            equity_curve.append({'date': rd, 'equity': equity, 'n': 0, 'ret': 0,
                                'btc': 0, 'ic': 0, 'avg_prob': 0, 'n_elig': len(cross)})
            continue

        # Select shorts
        selected, weights, n = select_shorts(cross, feat_cols, clf, rnk)
        syms = selected.index.tolist()
        avg_prob = selected['loser_prob'].mean()

        # Simulate
        try:
            rets_dict = simulate_period(panel, syms, rd, HOLDING_DAYS, STOP_PCT)
        except Exception:
            rets_dict = {s: 0.0 for s in syms}

        # Confidence-weighted return
        ret_values = np.array([rets_dict.get(s, 0.0) for s in syms])
        weighted_ret = np.sum(ret_values * weights)

        # Funding
        if not funding_df.empty:
            fm = funding_df[funding_df.index <= rd]['fundingRate'].tail(21).mean()
            daily_f = abs(fm) * 3
        else:
            daily_f = FUNDING_DAILY_DEFAULT
        weighted_ret -= daily_f * HOLDING_DAYS

        # Costs
        weighted_ret -= 2 * COST_PER_SIDE

        if np.isnan(weighted_ret) or np.isinf(weighted_ret):
            weighted_ret = 0.0

        equity *= (1 + weighted_ret)
        if equity <= 0 or np.isnan(equity):
            equity = max(1.0, equity if not np.isnan(equity) else 1.0)

        # Rank IC
        ic = 0.0
        try:
            all_rets = simulate_period(panel, cross.index.tolist(), rd, HOLDING_DAYS, 1.0)
            r_series = pd.Series(all_rets)
            p_series = cross['loser_prob'] if 'loser_prob' in cross.columns else pd.Series(0, index=cross.index)
            common = r_series.index.intersection(p_series.index)
            if len(common) >= 10:
                ic_val, _ = spearmanr(p_series.loc[common], -r_series.loc[common])
                ic = ic_val if not np.isnan(ic_val) else 0.0
        except Exception:
            pass

        btc_p = 0
        try:
            btc_p = panel.loc[(rd, 'BTCUSDT'), 'close']
            if hasattr(btc_p, 'iloc'):
                btc_p = btc_p.iloc[0]
        except Exception:
            pass

        equity_curve.append({'date': rd, 'equity': equity, 'n': n, 'ret': weighted_ret,
                            'btc': btc_p, 'ic': ic, 'avg_prob': avg_prob,
                            'n_elig': len(cross)})
        diagnostics.append({'date': rd, 'ic': ic, 'n': n, 'avg_prob': avg_prob,
                           'ret': weighted_ret, 'n_elig': len(cross)})

    # Results
    result = pd.DataFrame(equity_curve).set_index('date')
    result.to_parquet(os.path.join(RESULTS_DIR, 'ml_v5_result.parquet'))
    diag = pd.DataFrame(diagnostics).set_index('date')
    diag.to_parquet(os.path.join(RESULTS_DIR, 'ml_v5_diagnostics.parquet'))

    avg_imp = pd.Series(dtype=float)
    if all_imp:
        avg_imp = pd.concat(all_imp, axis=1).mean(axis=1).sort_values(ascending=False)
        avg_imp.to_json(os.path.join(RESULTS_DIR, 'ml_v5_importance.json'))

    eq = result['equity']
    n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_yrs) - 1 if eq.iloc[-1] > 0 else -1
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(365/HOLDING_DAYS) if rets.std() > 0 else 0
    ds = rets[rets < 0].std()
    sortino = rets.mean() / ds * np.sqrt(365/HOLDING_DAYS) if ds and ds > 0 else 0
    pk = eq.expanding().max()
    mdd = ((eq - pk) / pk).min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    yearly = {}
    for yr in sorted(eq.index.year.unique()):
        y = eq[eq.index.year == yr]
        if len(y) > 1:
            yearly[yr] = (y.iloc[-1] / y.iloc[0] - 1) * 100

    avg_ic = diag['ic'].mean() if len(diag) > 0 else 0
    ic_pos = (diag['ic'] > 0).mean() * 100 if len(diag) > 0 else 0

    print(f"\n{'='*70}")
    print(f"  ML v5 SHORT-ONLY IMPROVED")
    print(f"  BASELINE TO BEAT: Sharpe 1.36, +933%, Max DD -37.9%")
    print(f"{'='*70}")
    print(f"  Period:    {eq.index[0].date()} to {eq.index[-1].date()}")
    print(f"  $100k ->   ${eq.iloc[-1]:,.0f}")
    print(f"  CAGR:      {cagr*100:.1f}%")
    print(f"  Sharpe:    {sharpe:.2f}")
    print(f"  Sortino:   {sortino:.2f}")
    print(f"  Max DD:    {mdd*100:.1f}%")
    print(f"  Calmar:    {calmar:.2f}")
    for yr, r in yearly.items():
        print(f"  {yr}:      {r:+.1f}%")
    print(f"\n  ML DIAGNOSTICS:")
    print(f"  Avg IC:    {avg_ic:.4f}")
    print(f"  IC > 0:    {ic_pos:.0f}%")
    print(f"  Avg N:     {diag['n'].mean():.1f}")
    print(f"  Avg P:     {diag['avg_prob'].mean():.3f}")

    beat = "BEATS BASELINE" if sharpe > 1.36 else "DOES NOT BEAT"
    print(f"\n  >>> {beat} (Sharpe {sharpe:.2f} vs 1.36) <<<")

    if len(avg_imp) > 0:
        print(f"\n  TOP 10 FEATURES:")
        for f, v in avg_imp.head(10).items():
            print(f"    {f:25s}  {v:.4f}")
    print(f"{'='*70}")

    # Dashboard
    _build_dashboard(result, avg_imp, diag, sharpe, cagr, mdd)
    return result, avg_imp


def _build_dashboard(result, importance, diag, sharpe, cagr, mdd):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    eq = result['equity']
    fig = make_subplots(rows=4, cols=2, subplot_titles=[
        'Equity (log) — v5 vs Baseline vs BTC', 'Feature Importance (Classifier)',
        'Drawdown', 'Rank IC Over Time',
        'Annual Returns', 'Dynamic N Shorts',
        'Confidence P(loser)', 'Return Distribution',
    ], row_heights=[0.3, 0.25, 0.25, 0.2])

    # 1. Equity
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name='v5 Short Improved',
                             line=dict(color='#FF5722', width=3)), row=1, col=1)
    btc = result['btc'].replace(0, np.nan).dropna()
    if len(btc) > 0 and btc.iloc[0] > 0:
        fig.add_trace(go.Scatter(x=btc.index, y=INITIAL_CAPITAL*btc/btc.iloc[0],
                                 name='BTC B&H', line=dict(color='#FFD700', width=1.5, dash='dot')), row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="gray", row=1, col=1)

    # 2. Feature importance
    if len(importance) > 0:
        top = importance.head(15)[::-1]
        colors = ['#FF5722' if 'poly' in f else '#2196F3' if 'mom' in f else
                  '#FF9800' if any(x in f for x in ['amihud','spread','turn']) else '#4CAF50'
                  for f in top.index]
        fig.add_trace(go.Bar(y=top.index, x=top.values, orientation='h',
                             marker_color=colors, name='Importance'), row=1, col=2)

    # 3. DD
    pk = eq.expanding().max()
    dd = (eq-pk)/pk*100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy',
                             line=dict(color='#f44336', width=1), name='DD'), row=2, col=1)

    # 4. IC
    if len(diag) > 0:
        ic = diag['ic']
        fig.add_trace(go.Bar(x=ic.index, y=ic, name='IC',
                             marker_color=['#4CAF50' if v>0 else '#f44336' for v in ic],
                             opacity=0.5), row=2, col=2)
        fig.add_trace(go.Scatter(x=ic.rolling(6).mean().index, y=ic.rolling(6).mean(),
                                 name='IC MA6', line=dict(color='#FFD700', width=2)), row=2, col=2)
        fig.add_hline(y=0, line_color="gray", row=2, col=2)

    # 5. Annual
    yearly = {}
    for yr in sorted(eq.index.year.unique()):
        y = eq[eq.index.year == yr]
        if len(y) > 1: yearly[yr] = (y.iloc[-1]/y.iloc[0]-1)*100
    if yearly:
        fig.add_trace(go.Bar(x=[str(y) for y in yearly], y=list(yearly.values()),
                             marker_color=['#4CAF50' if v>0 else '#f44336' for v in yearly.values()],
                             text=[f'{v:+.0f}%' for v in yearly.values()], textposition='outside',
                             name='Annual'), row=3, col=1)

    # 6. Dynamic N
    if len(diag) > 0:
        fig.add_trace(go.Scatter(x=diag.index, y=diag['n'], name='N shorts',
                                 line=dict(color='#03A9F4', width=2)), row=3, col=2)

    # 7. Confidence
    if len(diag) > 0:
        fig.add_trace(go.Scatter(x=diag.index, y=diag['avg_prob'], name='P(loser)',
                                 line=dict(color='#FF9800', width=2)), row=4, col=1)

    # 8. Ret dist
    fig.add_trace(go.Histogram(x=result['ret']*100, nbinsx=40,
                               marker_color='#f44336', name='Returns'), row=4, col=2)

    n_yrs = (eq.index[-1]-eq.index[0]).days/365.25
    fig.update_layout(
        height=1800, template='plotly_dark',
        title_text=(f'ML v5 Short-Only Improved | Sharpe: {sharpe:.2f} | '
                    f'CAGR: {cagr*100:.1f}% | DD: {mdd*100:.1f}% | '
                    f'${INITIAL_CAPITAL/1000:.0f}k → ${eq.iloc[-1]/1000:.0f}k | '
                    f'{"BEATS" if sharpe > 1.36 else "MISS"} baseline 1.36'),
    )

    path = os.path.join(RESULTS_DIR, 'ml_v5_dashboard.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"Dashboard: {path}")
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    run()
