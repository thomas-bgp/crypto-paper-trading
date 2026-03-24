"""
ML v6 — SHORT-ONLY 5-DAY HOLD. Tests hypothesis that signal decays fast.
Identical to v5 baseline except HOLDING_DAYS=5 (was 14).
Period: 2025-01-01 to 2026-03-23 only.

Hypothesis: shorter hold captures alpha before mean reversion kicks in.
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
PURGE_DAYS = 8               # holding + small buffer
HOLDING_DAYS = 5              # ← KEY CHANGE: was 14
MIN_N = 3
MAX_N = 8
LOSER_THRESHOLD = 0.10
CONFIDENCE_GATE = 0.50
UNIVERSE_TOP = 50
STOP_PCT = 0.15
COST_PER_SIDE = 0.002
FUNDING_DAILY_DEFAULT = 0.001
INITIAL_CAPITAL = 100_000

# Backtest window
BT_START = pd.Timestamp('2025-01-01')
BT_END = pd.Timestamp('2026-03-23')

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

    clf = None
    try:
        clf = CatBoostClassifier(**CLASSIFIER_PARAMS)
        clf.fit(X, train_data['is_loser'].values)
    except Exception as e:
        print(f"    Clf fail: {e}")

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
    X = np.nan_to_num(cross[feat_cols].values, nan=0, posinf=0, neginf=0)

    loser_probs = np.full(len(X), 0.5)
    if clf is not None:
        try:
            loser_probs = clf.predict_proba(X)[:, 1]
        except Exception:
            pass

    cross = cross.copy()
    cross['loser_prob'] = loser_probs

    candidates = cross[cross['loser_prob'] >= CONFIDENCE_GATE]
    if len(candidates) < MIN_N:
        candidates = cross.nlargest(MIN_N, 'loser_prob')

    rank_scores = np.zeros(len(candidates))
    if rnk is not None:
        try:
            Xc = np.nan_to_num(candidates[feat_cols].values, nan=0, posinf=0, neginf=0)
            rank_scores = rnk.predict(Xc)
        except Exception:
            pass

    candidates = candidates.copy()
    candidates['rank_score'] = rank_scores
    n = max(MIN_N, min(len(candidates), MAX_N))
    selected = candidates.nsmallest(n, 'rank_score')

    probs = selected['loser_prob'].values
    weights = probs / probs.sum()
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
    print("  ML v6 — SHORT-ONLY 5-DAY HOLD")
    print(f"  Period: {BT_START.date()} to {BT_END.date()}")
    print(f"  Holding: {HOLDING_DAYS} days (baseline=14)")
    print("  Hypothesis: signal decays fast, shorter hold captures more alpha")
    print("=" * 70)

    panel, feat_cols = load_and_compute()
    dates = panel.index.get_level_values('date').unique().sort_values()

    # Rebalance dates within backtest window
    rebal_dates = []
    d = BT_START
    while d <= BT_END - pd.Timedelta(days=HOLDING_DAYS):
        nearest = dates[dates <= d]
        if len(nearest) > 0:
            rebal_dates.append(nearest[-1])
        d += pd.Timedelta(days=HOLDING_DAYS)
    rebal_dates = sorted(set(rebal_dates))
    print(f"Rebalances: {len(rebal_dates)} (every {HOLDING_DAYS} days)")

    # Also run the 14-day baseline on same period for fair comparison
    rebal_14 = []
    d14 = BT_START
    while d14 <= BT_END - pd.Timedelta(days=14):
        nearest = dates[dates <= d14]
        if len(nearest) > 0:
            rebal_14.append(nearest[-1])
        d14 += pd.Timedelta(days=14)
    rebal_14 = sorted(set(rebal_14))

    funding_df = pd.DataFrame()
    fr_path = os.path.join(DATA_DIR, 'BTCUSDT_funding.parquet')
    if os.path.exists(fr_path):
        funding_df = pd.read_parquet(fr_path)

    # ── Run both strategies ──
    results = {}
    for label, hold, rebals in [('5d_hold', HOLDING_DAYS, rebal_dates),
                                  ('14d_hold', 14, rebal_14)]:
        print(f"\n{'-'*50}")
        print(f"  Running: {label} ({len(rebals)} rebalances)")
        print(f"{'-'*50}")

        equity = INITIAL_CAPITAL
        equity_curve = []
        diagnostics = []
        all_imp = []
        clf = rnk = None

        # Retrain frequency: every ~56 days
        retrain_every = max(1, 56 // hold)

        for i, rd in enumerate(rebals):
            should_train = (i % retrain_every == 0) or i == 0
            if should_train:
                te = rd
                ts = te - pd.DateOffset(months=TRAIN_MONTHS)
                tmask = ((panel.index.get_level_values('date') >= ts) &
                         (panel.index.get_level_values('date') < te))
                purge = hold + 3  # hold + small buffer
                td = build_training_with_purge(panel, tmask, hold, purge)
                print(f"  [{i+1}/{len(rebals)}] {rd.date()} Train {len(td)} rows "
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
                                    'btc': 0, 'ic': 0, 'avg_prob': 0})
                continue

            if rd not in panel.index.get_level_values('date'):
                equity_curve.append({'date': rd, 'equity': equity, 'n': 0, 'ret': 0,
                                    'btc': 0, 'ic': 0, 'avg_prob': 0})
                continue

            cross = panel.loc[rd].copy()
            cross = cross.dropna(subset=['mom_14', 'vol_avg_28'])
            cross = cross[cross['vol_avg_28'] > 0]
            cross = cross.nlargest(UNIVERSE_TOP, 'vol_avg_28')

            if len(cross) < 10:
                equity_curve.append({'date': rd, 'equity': equity, 'n': 0, 'ret': 0,
                                    'btc': 0, 'ic': 0, 'avg_prob': 0})
                continue

            selected, weights, n = select_shorts(cross, feat_cols, clf, rnk)
            syms = selected.index.tolist()
            avg_prob = selected['loser_prob'].mean()

            try:
                rets_dict = simulate_period(panel, syms, rd, hold, STOP_PCT)
            except Exception:
                rets_dict = {s: 0.0 for s in syms}

            ret_values = np.array([rets_dict.get(s, 0.0) for s in syms])
            weighted_ret = np.sum(ret_values * weights)

            # Funding cost
            if not funding_df.empty:
                fm = funding_df[funding_df.index <= rd]['fundingRate'].tail(21).mean()
                daily_f = abs(fm) * 3
            else:
                daily_f = FUNDING_DAILY_DEFAULT
            weighted_ret -= daily_f * hold

            # Trading costs
            weighted_ret -= 2 * COST_PER_SIDE

            if np.isnan(weighted_ret) or np.isinf(weighted_ret):
                weighted_ret = 0.0

            equity *= (1 + weighted_ret)
            if equity <= 0 or np.isnan(equity):
                equity = max(1.0, equity if not np.isnan(equity) else 1.0)

            # IC (lightweight: use forward returns from panel instead of full simulation)
            ic = 0.0
            try:
                fwd_col = f'fwd_{min(hold, 14)}'
                if fwd_col in cross.columns:
                    valid = cross.dropna(subset=[fwd_col, 'loser_prob'])
                    if len(valid) >= 10:
                        ic_val, _ = spearmanr(valid['loser_prob'], -valid[fwd_col])
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
                                'btc': btc_p, 'ic': ic, 'avg_prob': avg_prob})
            diagnostics.append({'date': rd, 'ic': ic, 'n': n, 'avg_prob': avg_prob,
                               'ret': weighted_ret})

        result = pd.DataFrame(equity_curve).set_index('date')
        diag = pd.DataFrame(diagnostics).set_index('date') if diagnostics else pd.DataFrame()
        results[label] = {'result': result, 'diag': diag, 'importance': all_imp}

    # ═══ Print comparison ═══
    print_comparison(results)
    build_dashboard(results, panel)


def build_training_with_purge(panel, train_mask, hold_days, purge_days):
    """Build training data with correct purge for the given holding period."""
    train = panel[train_mask].copy()
    g = train.groupby(level='symbol')
    train['fwd'] = g['close'].transform(lambda x: x.pct_change(hold_days).shift(-hold_days))
    cutoff = train.index.get_level_values('date').max() - pd.Timedelta(days=purge_days)
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


def print_comparison(results):
    print(f"\n{'='*70}")
    print(f"  COMPARISON: 5-DAY vs 14-DAY HOLD")
    print(f"  Period: {BT_START.date()} to {BT_END.date()}")
    print(f"{'='*70}")

    for label, data in results.items():
        eq = data['result']['equity']
        if len(eq) < 2:
            print(f"  {label}: insufficient data")
            continue

        hold = 5 if '5d' in label else 14
        n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_yrs) - 1 if eq.iloc[-1] > 0 else -1
        rets = eq.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(365/hold) if rets.std() > 0 else 0
        ds = rets[rets < 0].std()
        sortino = rets.mean() / ds * np.sqrt(365/hold) if ds and ds > 0 else 0
        pk = eq.expanding().max()
        mdd = ((eq - pk) / pk).min()
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        n_trades = len(rets)
        win_rate = (rets > 0).mean() * 100

        diag = data['diag']
        avg_ic = diag['ic'].mean() if len(diag) > 0 else 0

        print(f"\n  ── {label.upper()} ──")
        print(f"  $100k →    ${eq.iloc[-1]:,.0f}  ({total_ret:+.1f}%)")
        print(f"  CAGR:      {cagr*100:.1f}%")
        print(f"  Sharpe:    {sharpe:.2f}")
        print(f"  Sortino:   {sortino:.2f}")
        print(f"  Max DD:    {mdd*100:.1f}%")
        print(f"  Calmar:    {calmar:.2f}")
        print(f"  Win Rate:  {win_rate:.0f}% ({n_trades} periods)")
        print(f"  Avg IC:    {avg_ic:.4f}")

    print(f"\n{'='*70}")


def build_dashboard(results, panel):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=3, cols=2, subplot_titles=[
        'Equity Curve (log scale)', 'Period Returns Distribution',
        'Drawdown Comparison', 'Rank IC Over Time (5d)',
        'Cumulative Return (%)', 'Win Rate Rolling (10 periods)',
    ], row_heights=[0.35, 0.35, 0.30])

    colors = {'5d_hold': '#FF5722', '14d_hold': '#2196F3'}

    for label, data in results.items():
        eq = data['result']['equity']
        if len(eq) < 2:
            continue
        color = colors[label]
        hold = 5 if '5d' in label else 14

        # 1. Equity (log)
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=f'{label}',
                                 line=dict(color=color, width=2.5)), row=1, col=1)

        # 2. Return distribution
        rets = data['result']['ret'] * 100
        fig.add_trace(go.Histogram(x=rets, name=f'{label} rets', opacity=0.6,
                                    marker_color=color, nbinsx=30), row=1, col=2)

        # 3. Drawdown
        pk = eq.expanding().max()
        dd = (eq - pk) / pk * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f'{label} DD',
                                 fill='tozeroy', line=dict(color=color, width=1),
                                 opacity=0.5), row=2, col=1)

        # 4. IC (only for 5d)
        if '5d' in label and len(data['diag']) > 0:
            ic = data['diag']['ic']
            fig.add_trace(go.Bar(x=ic.index, y=ic, name='IC (5d)',
                                 marker_color=['#4CAF50' if v > 0 else '#f44336' for v in ic],
                                 opacity=0.5), row=2, col=2)
            if len(ic) > 5:
                fig.add_trace(go.Scatter(x=ic.rolling(5).mean().index,
                                         y=ic.rolling(5).mean(),
                                         name='IC MA5', line=dict(color='#FFD700', width=2)),
                             row=2, col=2)

        # 5. Cumulative return %
        cum_ret = (eq / eq.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name=f'{label} cum%',
                                 line=dict(color=color, width=2)), row=3, col=1)

        # 6. Rolling win rate
        rets_s = eq.pct_change().dropna()
        if len(rets_s) > 10:
            rolling_wr = (rets_s > 0).rolling(10).mean() * 100
            fig.add_trace(go.Scatter(x=rolling_wr.index, y=rolling_wr,
                                     name=f'{label} WR10', line=dict(color=color, width=2)),
                         row=3, col=2)

    # BTC benchmark
    try:
        btc_data = results['5d_hold']['result']['btc'].replace(0, np.nan).dropna()
        if len(btc_data) > 0 and btc_data.iloc[0] > 0:
            btc_eq = INITIAL_CAPITAL * btc_data / btc_data.iloc[0]
            fig.add_trace(go.Scatter(x=btc_eq.index, y=btc_eq, name='BTC B&H',
                                     line=dict(color='#FFD700', width=1.5, dash='dot')),
                         row=1, col=1)
    except Exception:
        pass

    fig.update_yaxes(type='log', row=1, col=1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_color="gray", row=2, col=2)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=2)

    # Summary stats for title
    stats = {}
    for label, data in results.items():
        eq = data['result']['equity']
        if len(eq) < 2:
            continue
        hold = 5 if '5d' in label else 14
        rets = eq.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(365/hold) if rets.std() > 0 else 0
        total = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        pk = eq.expanding().max()
        mdd = ((eq - pk) / pk).min() * 100
        stats[label] = (sharpe, total, mdd)

    title_parts = []
    for label, (s, t, m) in stats.items():
        title_parts.append(f"{label}: Sharpe {s:.2f}, {t:+.0f}%, DD {m:.0f}%")

    fig.update_layout(
        height=1400, template='plotly_dark',
        title_text=f"5-Day vs 14-Day Hold | {' | '.join(title_parts)}",
        showlegend=True,
    )

    path = os.path.join(RESULTS_DIR, 'ml_v6_hold5_dashboard.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")

    # Save results
    for label, data in results.items():
        data['result'].to_parquet(os.path.join(RESULTS_DIR, f'ml_v6_{label}_result.parquet'))

    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    run()
