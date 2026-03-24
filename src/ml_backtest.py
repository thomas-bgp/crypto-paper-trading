"""
ML Cross-Sectional Momentum Model — Walk-Forward Backtest.
Models: LGBMRanker (LambdaMART), LightGBM Regressor, Elastic Net.
Ensemble of all three. Walk-forward: train 12mo rolling, predict 1mo ahead.
"""
import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import shap

from ml_features import build_ml_dataset, get_feature_columns

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Config ───
TRAIN_MONTHS = 12       # rolling training window
HOLDING_DAYS = 14       # rebalance every 14 days
TOP_N = 8               # long top 8 + short bottom 8
UNIVERSE_TOP = 50       # top 50 by volume
STOP_PCT = 0.15         # trailing stop approximation
COST_PER_SIDE = 0.00125
FUNDING_DAILY = 0.00037
INITIAL_CAPITAL = 100_000
TARGET_COL = 'fwd_14'
TARGET_RANK_COL = 'fwd_14_rank'

LGBM_PARAMS = {
    'n_estimators': 200,
    'num_leaves': 15,
    'max_depth': 4,
    'learning_rate': 0.05,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
}


def prepare_cross_section(panel, date, feat_cols):
    """Get one cross-section for a given date, filtered and cleaned."""
    if date not in panel.index.get_level_values('date'):
        return None
    cross = panel.loc[date].copy()
    cross = cross.dropna(subset=[TARGET_COL, 'vol_avg_28'])
    cross = cross[cross['vol_avg_28'] > 0]
    # Top N by volume
    cross = cross.nlargest(UNIVERSE_TOP, 'vol_avg_28')
    if len(cross) < 15:
        return None
    return cross


def train_models(train_data, feat_cols):
    """Train LGBMRanker, LGBMRegressor, ElasticNet on training panel."""
    # Clean
    train_data = train_data.dropna(subset=feat_cols + [TARGET_COL])
    if len(train_data) < 100:
        return None, None, None

    X = train_data[feat_cols].values
    y_ret = train_data[TARGET_COL].values
    # Winsorize returns at 1st/99th percentile
    p1, p99 = np.percentile(y_ret, [1, 99])
    y_ret = np.clip(y_ret, p1, p99)

    # Cross-sectional rank target for ranker
    y_rank = train_data.groupby(level='date')[TARGET_COL].rank(pct=True).values
    # Relevance labels (0-4 quintiles) for LGBMRanker
    y_label = (y_rank * 4.99).astype(int).clip(0, 4)

    # Query groups for ranker (each date is a query)
    dates = train_data.index.get_level_values('date')
    unique_dates = dates.unique()
    group_sizes = [int((dates == d).sum()) for d in unique_dates]

    # Handle NaN/inf in features
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # ── 1. LGBMRanker ──
    ranker = None
    try:
        ranker = LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            label_gain=[0, 1, 3, 7, 15],  # relevance gains for labels 0-4
            **LGBM_PARAMS,
        )
        ranker.fit(X, y_label, group=group_sizes)
    except Exception as e:
        print(f"    Ranker failed: {e}")
        ranker = None

    # ── 2. LGBMRegressor ──
    regressor = None
    try:
        regressor = LGBMRegressor(**LGBM_PARAMS)
        regressor.fit(X, y_ret)
    except Exception as e:
        print(f"    Regressor failed: {e}")

    # ── 3. ElasticNet ──
    enet = None
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=2000, n_jobs=-1)
        enet.fit(X_scaled, y_ret)
        enet._scaler = scaler  # attach for prediction
    except Exception as e:
        print(f"    ElasticNet failed: {e}")

    return ranker, regressor, enet


def predict_scores(cross, feat_cols, ranker, regressor, enet):
    """Ensemble prediction: average normalized scores from 3 models."""
    X = cross[feat_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    scores = []

    if ranker is not None:
        try:
            s = ranker.predict(X)
            s = (s - s.mean()) / (s.std() + 1e-10)
            scores.append(s)
        except Exception:
            pass

    if regressor is not None:
        try:
            s = regressor.predict(X)
            s = (s - s.mean()) / (s.std() + 1e-10)
            scores.append(s)
        except Exception:
            pass

    if enet is not None:
        try:
            X_scaled = enet._scaler.transform(X)
            s = enet.predict(X_scaled)
            s = (s - s.mean()) / (s.std() + 1e-10)
            scores.append(s)
        except Exception:
            pass

    if not scores:
        return None

    # Ensemble: average of normalized scores
    ensemble = np.mean(scores, axis=0)
    return ensemble


def run_ml_backtest():
    """Walk-forward ML backtest."""
    print("=" * 70)
    print("  ML CROSS-SECTIONAL MOMENTUM BACKTEST")
    print("  Models: LGBMRanker + LGBMRegressor + ElasticNet (ensemble)")
    print("=" * 70)

    # Load & compute features
    panel, feat_cols = build_ml_dataset()
    print(f"\nFeatures ({len(feat_cols)}): {feat_cols}")

    dates = panel.index.get_level_values('date').unique().sort_values()
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

    # Walk-forward dates: start after TRAIN_MONTHS warmup
    start_date = dates[0] + pd.DateOffset(months=TRAIN_MONTHS + 3)
    rebal_dates = pd.date_range(start_date, dates[-1], freq=f'{HOLDING_DAYS}D')
    rebal_dates = rebal_dates[rebal_dates.isin(dates) | True]  # keep all
    # Find nearest available date for each rebal date
    rebal_dates_actual = []
    for rd in rebal_dates:
        nearest = dates[dates <= rd]
        if len(nearest) > 0:
            rebal_dates_actual.append(nearest[-1])
    rebal_dates_actual = sorted(set(rebal_dates_actual))

    print(f"Rebalance dates: {len(rebal_dates_actual)} ({rebal_dates_actual[0].date()} to {rebal_dates_actual[-1].date()})")

    # ─── Main walk-forward loop ───
    equity = INITIAL_CAPITAL
    equity_curve = []
    all_importances = []
    model_scores = {'ranker': 0, 'regressor': 0, 'enet': 0}
    train_count = 0

    for i, rebal_date in enumerate(rebal_dates_actual):
        # ── Train window ──
        train_end = rebal_date
        train_start = train_end - pd.DateOffset(months=TRAIN_MONTHS)
        train_mask = (panel.index.get_level_values('date') >= train_start) & \
                     (panel.index.get_level_values('date') < train_end)
        train_data = panel[train_mask]

        # Filter train data to universe (top by volume each month in train)
        # Simplified: just use volume filter
        train_data = train_data[train_data['vol_avg_28'] > train_data.groupby(level='date')['vol_avg_28'].transform(
            lambda x: x.quantile(0.3))]

        # ── Retrain every 2 months (save compute) ──
        should_train = (i % (60 // HOLDING_DAYS) == 0) or i == 0
        if should_train:
            print(f"  [{i+1}/{len(rebal_dates_actual)}] {rebal_date.date()} — Training on {train_start.date()} to {train_end.date()} ({len(train_data)} rows)...", flush=True)
            ranker, regressor, enet = train_models(train_data, feat_cols)
            train_count += 1

            # Feature importance (from regressor)
            if regressor is not None:
                imp = pd.Series(regressor.feature_importances_, index=feat_cols)
                imp = imp / (imp.sum() + 1e-10)
                all_importances.append(imp)

        # ── Predict on current cross-section ──
        cross = prepare_cross_section(panel, rebal_date, feat_cols)
        if cross is None or ranker is None:
            equity_curve.append({'date': rebal_date, 'equity': equity, 'n_pos': 0, 'model': 'skip'})
            continue

        scores = predict_scores(cross, feat_cols, ranker, regressor, enet)
        if scores is None:
            equity_curve.append({'date': rebal_date, 'equity': equity, 'n_pos': 0, 'model': 'fail'})
            continue

        cross = cross.copy()
        cross['ml_score'] = scores
        cross = cross.sort_values('ml_score', ascending=False)

        # ── Portfolio construction ──
        longs = cross.head(TOP_N)
        shorts = cross.tail(TOP_N)

        # Long return
        long_ret = longs[TARGET_COL].clip(lower=-STOP_PCT).mean()
        # Short return (profit from shorting)
        short_ret = -shorts[TARGET_COL].clip(upper=STOP_PCT).mean()
        short_ret -= FUNDING_DAILY * HOLDING_DAYS

        total_ret = (long_ret + short_ret) / 2
        total_ret -= 2 * COST_PER_SIDE  # costs

        equity *= (1 + total_ret)
        if equity <= 0:
            equity = 1

        # Get BTC price for comparison
        btc_price = 0
        if 'BTCUSDT' in panel.index.get_level_values('symbol').unique():
            btc_mask = (panel.index.get_level_values('date') == rebal_date) & \
                       (panel.index.get_level_values('symbol') == 'BTCUSDT')
            if btc_mask.any():
                btc_price = panel.loc[btc_mask, 'close'].iloc[0]

        equity_curve.append({
            'date': rebal_date,
            'equity': equity,
            'n_pos': TOP_N * 2,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'total_ret': total_ret,
            'btc_close': btc_price,
            'top_long': ', '.join(longs.index.get_level_values('symbol')[:3].tolist()) if hasattr(longs.index, 'get_level_values') else '',
        })

    # ─── Results ───
    result = pd.DataFrame(equity_curve).set_index('date')
    result.to_parquet(os.path.join(RESULTS_DIR, 'ml_backtest_result.parquet'))

    # Feature importance
    if all_importances:
        avg_imp = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        avg_imp.to_json(os.path.join(RESULTS_DIR, 'ml_feature_importance.json'))
    else:
        avg_imp = pd.Series(dtype=float)

    # Metrics
    eq = result['equity']
    rets = eq.pct_change().dropna()
    n_days = (result.index[-1] - result.index[0]).days
    n_years = n_days / 365.25

    if n_years > 0 and eq.iloc[-1] > 0:
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_years) - 1
        sharpe = rets.mean() / rets.std() * np.sqrt(365/HOLDING_DAYS) if rets.std() > 0 else 0
        downside = rets[rets < 0].std()
        sortino = rets.mean() / downside * np.sqrt(365/HOLDING_DAYS) if downside and downside > 0 else 0
        peak = eq.expanding().max()
        max_dd = ((eq - peak) / peak).min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    else:
        cagr = sharpe = sortino = max_dd = calmar = 0

    # Yearly
    yearly = {}
    for year in sorted(result.index.year.unique()):
        yr = result[result.index.year == year]['equity']
        if len(yr) > 1:
            yearly[year] = (yr.iloc[-1] / yr.iloc[0] - 1) * 100

    print(f"\n{'='*70}")
    print(f"  ML BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Period:       {result.index[0].date()} to {result.index[-1].date()}")
    print(f"  Models trained: {train_count}")
    print(f"  Initial:      ${INITIAL_CAPITAL:,.0f}")
    print(f"  Final:        ${eq.iloc[-1]:,.0f}")
    print(f"  CAGR:         {cagr*100:.1f}%")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Sortino:      {sortino:.2f}")
    print(f"  Max DD:       {max_dd*100:.1f}%")
    print(f"  Calmar:       {calmar:.2f}")
    for yr, ret in yearly.items():
        print(f"  {yr}:         {ret:+.1f}%")

    if len(avg_imp) > 0:
        print(f"\n  TOP 20 FEATURES BY IMPORTANCE:")
        for feat, imp in avg_imp.head(20).items():
            print(f"    {feat:25s}  {imp:.4f}")

    print(f"{'='*70}")

    # ─── SHAP analysis on last model ───
    if regressor is not None and cross is not None:
        print("\nComputing SHAP values (last cross-section)...")
        try:
            X_last = cross[feat_cols].values
            X_last = np.nan_to_num(X_last, nan=0, posinf=0, neginf=0)
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer.shap_values(X_last)
            shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_cols).sort_values(ascending=False)
            shap_imp.to_json(os.path.join(RESULTS_DIR, 'ml_shap_importance.json'))
            print("  TOP 20 SHAP FEATURES:")
            for feat, val in shap_imp.head(20).items():
                print(f"    {feat:25s}  {val:.6f}")
        except Exception as e:
            print(f"  SHAP failed: {e}")

    return result, avg_imp


def build_ml_dashboard(result, importance):
    """Build equity curve + importance dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    eq = result['equity']

    # ── Equity curve ──
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Equity Curve (ML Ensemble)', 'Feature Importance (LightGBM)',
                        'Drawdown', 'SHAP Importance',
                        'Annual Returns', 'Per-Period Returns Distribution'],
        row_heights=[0.4, 0.3, 0.3],
        specs=[[{}, {}], [{}, {}], [{}, {}]],
    )

    # 1. Equity
    fig.add_trace(go.Scatter(
        x=result.index, y=eq, name='ML Ensemble',
        line=dict(color='#FF5722', width=2.5),
    ), row=1, col=1)

    # Add BTC benchmark
    if 'btc_close' in result.columns:
        btc = result['btc_close'].dropna()
        if len(btc) > 0 and btc.iloc[0] > 0:
            btc_eq = INITIAL_CAPITAL * btc / btc.iloc[0]
            fig.add_trace(go.Scatter(
                x=btc_eq.index, y=btc_eq, name='BTC Buy&Hold',
                line=dict(color='#FFD700', width=1.5, dash='dot'),
            ), row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=1)

    # 2. Feature importance (LightGBM)
    if len(importance) > 0:
        top20 = importance.head(20)[::-1]
        fig.add_trace(go.Bar(
            y=top20.index, x=top20.values, orientation='h',
            marker_color='#4CAF50', name='LightGBM Importance',
        ), row=1, col=2)

    # 3. Drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, fill='tozeroy', name='Drawdown',
        line=dict(color='#f44336', width=1),
    ), row=2, col=1)

    # 4. SHAP importance
    shap_path = os.path.join(RESULTS_DIR, 'ml_shap_importance.json')
    if os.path.exists(shap_path):
        shap_imp = pd.read_json(shap_path, typ='series').sort_values(ascending=True).tail(20)
        fig.add_trace(go.Bar(
            y=shap_imp.index, x=shap_imp.values, orientation='h',
            marker_color='#FF9800', name='SHAP Importance',
        ), row=2, col=2)

    # 5. Annual returns
    yearly = {}
    for year in sorted(result.index.year.unique()):
        yr = eq[eq.index.year == year]
        if len(yr) > 1:
            yearly[year] = (yr.iloc[-1] / yr.iloc[0] - 1) * 100
    if yearly:
        colors = ['#4CAF50' if v > 0 else '#f44336' for v in yearly.values()]
        fig.add_trace(go.Bar(
            x=[str(y) for y in yearly.keys()], y=list(yearly.values()),
            marker_color=colors, name='Annual Return',
            text=[f'{v:+.0f}%' for v in yearly.values()],
            textposition='outside',
        ), row=3, col=1)

    # 6. Return distribution
    rets = eq.pct_change().dropna() * 100
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=50, name='Period Returns',
        marker_color='#2196F3', opacity=0.7,
    ), row=3, col=2)
    fig.add_vline(x=0, line_color='white', row=3, col=2)

    fig.update_layout(
        height=1200, template='plotly_dark',
        title_text='ML Cross-Sectional Momentum — Walk-Forward Backtest',
        showlegend=True,
    )

    html_path = os.path.join(RESULTS_DIR, 'ml_dashboard.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"Dashboard saved to: {html_path}")
    return html_path


if __name__ == '__main__':
    result, importance = run_ml_backtest()
    path = build_ml_dashboard(result, importance)

    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')
