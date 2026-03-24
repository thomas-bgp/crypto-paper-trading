"""
Decile Calibration Analysis — Cross-Factor Momentum Model
==========================================================
Purpose: Check if the model's predicted scores have monotonic relationship
with realized forward returns. NO backtest, NO equity curve.

For each walk-forward window:
  1. Train model on past data
  2. Predict scores on the next cross-section
  3. Bin predictions into deciles (10 groups)
  4. Record realized forward return per decile

Output: calibration plot + statistics.
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMRanker
from catboost import CatBoostRanker
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Config ───
TRAIN_MONTHS = 12
PURGE_DAYS = 21          # gap between train end and test start
HOLDING_DAYS = 14        # forward return horizon
REBAL_EVERY = 14         # predict every N days
MIN_COINS_PER_DATE = 15  # need enough for deciles
N_DECILES = 10
VOL_FLOOR_QUANTILE = 0.2  # drop bottom 20% by volume

# LightGBM Ranker params
LGBM_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'num_leaves': 15,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

# CatBoost params for comparison
CATBOOST_PARAMS = {
    'loss_function': 'YetiRank',
    'iterations': 200,
    'depth': 4,
    'learning_rate': 0.05,
    'l2_leaf_reg': 5.0,
    'random_strength': 2.0,
    'bagging_temperature': 1.0,
    'border_count': 64,
    'verbose': 0,
    'random_seed': 42,
    'task_type': 'CPU',
}


# ════════════════════════════════════════════
# DATA LOADING (reuse feature_engine)
# ════════════════════════════════════════════

def load_panel():
    """Load panel using feature_engine.py"""
    from feature_engine import build_feature_matrix
    panel, _ = build_feature_matrix(resample='1D', min_candles=250)
    return panel


def get_feature_cols(panel):
    """Select feature columns — exclude targets, OHLCV, metadata."""
    exclude_prefixes = ('fwd_',)
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
# TARGET COMPUTATION
# ════════════════════════════════════════════

def compute_forward_return(panel, horizon=HOLDING_DAYS):
    """Compute forward return for each (date, symbol)."""
    col = f'fwd_ret_{horizon}d'
    if col not in panel.columns:
        panel[col] = panel.groupby(level='symbol')['close'].transform(
            lambda x: x.pct_change(horizon).shift(-horizon)
        )
    return col


def make_target_rank(df, target_col, n_bins=5):
    """Cross-sectional rank target (0 to n_bins-1)."""
    return df.groupby(level='date')[target_col].transform(
        lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop')
        if len(x.dropna()) >= n_bins else pd.Series(np.nan, index=x.index)
    ).fillna(n_bins // 2).astype(int)


# ════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════

def train_lgbm(X_train, y_train, groups):
    """Train LGBMRanker."""
    model = LGBMRanker(**LGBM_PARAMS)
    model.fit(X_train, y_train, group=groups)
    return model


def train_catboost(X_train, y_train, group_ids):
    """Train CatBoost Ranker."""
    model = CatBoostRanker(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, group_id=group_ids)
    return model


def predict_score(model, X, model_type='lgbm'):
    """Predict and normalize scores."""
    scores = model.predict(X)
    # Z-score normalize
    std = scores.std()
    if std > 1e-10:
        scores = (scores - scores.mean()) / std
    return scores


# ════════════════════════════════════════════
# WALK-FORWARD DECILE ANALYSIS
# ════════════════════════════════════════════

def run_decile_analysis(panel, feat_cols, model_type='lgbm'):
    """
    Walk-forward:
      - Train on [t - TRAIN_MONTHS, t - PURGE_DAYS]
      - Predict on date t
      - Bin into deciles
      - Record realized fwd return per decile
    """
    target_col = compute_forward_return(panel, HOLDING_DAYS)
    all_dates = panel.index.get_level_values('date').unique().sort_values()

    # Start after enough training data
    start_date = all_dates[0] + pd.DateOffset(months=TRAIN_MONTHS + 1)
    end_date = all_dates[-1] - pd.Timedelta(days=HOLDING_DAYS + 5)

    # Rebalance dates
    rebal_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    rebal_dates = rebal_dates[::REBAL_EVERY]
    print(f"Decile analysis: {len(rebal_dates)} prediction dates "
          f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    all_records = []       # per-coin per-date records
    period_stats = []      # per-date IC stats
    all_importances = []   # feature importance per retrain
    model = None
    last_train_date = None
    retrain_every = 60     # retrain every ~60 days
    available_feats = [f for f in feat_cols if f in panel.columns]

    for i, pred_date in enumerate(rebal_dates):
        # ── Should retrain? ──
        should_train = (
            model is None or
            last_train_date is None or
            (pred_date - last_train_date).days >= retrain_every
        )

        if should_train:
            train_end = pred_date - pd.Timedelta(days=PURGE_DAYS)
            train_start = train_end - pd.DateOffset(months=TRAIN_MONTHS)

            train_mask = (
                (panel.index.get_level_values('date') >= train_start) &
                (panel.index.get_level_values('date') <= train_end)
            )
            train_data = panel[train_mask].copy()

            # Volume filter on training data
            if 'vol_ratio_7d' in train_data.columns:
                pass  # keep all for training
            # Need forward returns as target
            train_data = train_data.dropna(subset=[target_col])

            # Compute ranked target
            train_data['target_rank'] = make_target_rank(train_data, target_col, n_bins=5)

            # Filter features
            available_feats = [f for f in feat_cols if f in train_data.columns]
            X_train = train_data[available_feats].values
            y_train = train_data['target_rank'].values
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

            # Group IDs for ranking
            train_dates_idx = train_data.index.get_level_values('date')
            group_codes = pd.Categorical(train_dates_idx).codes

            if model_type == 'lgbm':
                # LGBMRanker needs group sizes (count per query)
                group_sizes = pd.Series(group_codes).value_counts().sort_index().values
                try:
                    model = train_lgbm(X_train, y_train, group_sizes)
                except Exception as e:
                    print(f"  Train failed at {pred_date.date()}: {e}")
                    continue
            else:
                try:
                    model = train_catboost(X_train, y_train, group_codes)
                except Exception as e:
                    print(f"  Train failed at {pred_date.date()}: {e}")
                    continue

            # Collect feature importance
            try:
                if model_type == 'lgbm':
                    imp = pd.Series(model.feature_importances_, index=available_feats)
                else:
                    imp = pd.Series(
                        model.get_feature_importance(type='PredictionValuesChange'),
                        index=available_feats)
                imp = imp / (imp.sum() + 1e-10)
                all_importances.append(imp)
            except Exception:
                pass

            last_train_date = pred_date
            if i == 0 or (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(rebal_dates)}] Trained on "
                      f"{train_start.date()} to {train_end.date()} "
                      f"({len(train_data)} rows)")

        if model is None:
            continue

        # ── Predict on cross-section at pred_date ──
        if pred_date not in panel.index.get_level_values('date'):
            continue

        cross = panel.loc[pred_date].copy()
        cross = cross.dropna(subset=['close'])

        # Volume filter: drop bottom quantile
        if 'turnover_28d' in cross.columns:
            vol_thresh = cross['turnover_28d'].quantile(VOL_FLOOR_QUANTILE)
            cross = cross[cross['turnover_28d'] >= vol_thresh]

        if len(cross) < MIN_COINS_PER_DATE:
            continue

        available_feats = [f for f in feat_cols if f in cross.columns]
        X_pred = cross[available_feats].values
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

        scores = predict_score(model, X_pred, model_type)
        cross = cross.copy()
        cross['pred_score'] = scores

        # Forward return (realized)
        fwd_col = target_col
        if fwd_col not in cross.columns:
            continue
        cross = cross.dropna(subset=[fwd_col])
        if len(cross) < MIN_COINS_PER_DATE:
            continue

        # ── Bin into deciles ──
        try:
            cross['decile'] = pd.qcut(
                cross['pred_score'], N_DECILES,
                labels=range(1, N_DECILES + 1),
                duplicates='drop'
            ).astype(int)
        except ValueError:
            # Not enough unique values for N deciles
            cross['decile'] = pd.cut(
                cross['pred_score'].rank(method='first'),
                bins=N_DECILES, labels=range(1, N_DECILES + 1)
            ).astype(int)

        # Record per-coin
        syms = cross.index if cross.index.name == 'symbol' else cross.index.get_level_values('symbol') if 'symbol' in cross.index.names else cross.index
        for idx in range(len(cross)):
            row = cross.iloc[idx]
            all_records.append({
                'date': pred_date,
                'symbol': syms[idx] if hasattr(syms, '__getitem__') else str(idx),
                'pred_score': row['pred_score'],
                'decile': row['decile'],
                'fwd_return': row[fwd_col],
            })

        # Period IC
        ic, ic_p = spearmanr(cross['pred_score'].values, cross[fwd_col].values)
        period_stats.append({
            'date': pred_date,
            'rank_ic': ic,
            'rank_ic_p': ic_p,
            'n_coins': len(cross),
            'top_decile_ret': cross[cross['decile'] == N_DECILES][fwd_col].mean(),
            'bot_decile_ret': cross[cross['decile'] == 1][fwd_col].mean(),
            'spread': (cross[cross['decile'] == N_DECILES][fwd_col].mean() -
                       cross[cross['decile'] == 1][fwd_col].mean()),
        })

    records_df = pd.DataFrame(all_records)
    stats_df = pd.DataFrame(period_stats)

    if stats_df.empty:
        print("ERROR: No prediction dates produced results!")
        return records_df, stats_df

    stats_df = stats_df.set_index('date')

    print(f"\n{'='*60}")
    print(f"  DECILE CALIBRATION — {model_type.upper()}")
    print(f"{'='*60}")
    print(f"  Periods: {len(stats_df)}")
    print(f"  Avg Rank IC: {stats_df['rank_ic'].mean():.4f}")
    print(f"  IC > 0: {(stats_df['rank_ic'] > 0).mean()*100:.1f}%")
    print(f"  Avg Top Decile Return: {stats_df['top_decile_ret'].mean()*100:.2f}%")
    print(f"  Avg Bot Decile Return: {stats_df['bot_decile_ret'].mean()*100:.2f}%")
    print(f"  Avg L/S Spread: {stats_df['spread'].mean()*100:.2f}%")
    print(f"{'='*60}")

    # Aggregate feature importance
    avg_importance = pd.Series(dtype=float)
    if all_importances:
        avg_importance = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\n  TOP 20 FEATURES ({model_type.upper()}):")
        for f, v in avg_importance.head(20).items():
            print(f"    {f:30s}  {v:.4f}")

    return records_df, stats_df, avg_importance


# ════════════════════════════════════════════
# CALIBRATION PLOTS
# ════════════════════════════════════════════

def build_calibration_dashboard(records_df, stats_df, model_type='lgbm',
                                importance=None):
    """Build comprehensive decile calibration dashboard."""

    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            'Avg Forward Return by Decile (All Periods)',
            'Feature Importance (Top 25)',
            'Rank IC Over Time',
            'IC Distribution',
            'Decile Return Heatmap (by Period)',
            'Cumulative L/S Spread',
            'Top vs Bottom Decile Return Over Time',
            'Return Distribution per Decile (Box)',
            'Hit Rate by Decile (% Positive Return)',
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

    # ═══ 1. Main calibration bar chart: avg return per decile ═══
    decile_stats = records_df.groupby('decile')['fwd_return'].agg(['mean', 'std', 'count'])
    decile_stats['se'] = decile_stats['std'] / np.sqrt(decile_stats['count'])
    decile_stats['mean_pct'] = decile_stats['mean'] * 100

    # Color: gradient from red (low decile) to green (high decile)
    n = len(decile_stats)
    colors = [f'rgb({int(255*(1-i/(n-1)))}, {int(200*i/(n-1))}, 80)' for i in range(n)]

    fig.add_trace(go.Bar(
        x=[f'D{d}' for d in decile_stats.index],
        y=decile_stats['mean_pct'],
        error_y=dict(type='data', array=decile_stats['se'] * 100 * 1.96, visible=True),
        marker_color=colors,
        text=[f'{v:.2f}%' for v in decile_stats['mean_pct']],
        textposition='outside',
        name='Avg Fwd Return',
        hovertemplate='Decile %{x}: %{y:.2f}% ± %{error_y.array:.2f}%<extra></extra>',
    ), row=1, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=1, col=1)

    # Monotonicity score
    returns_by_decile = decile_stats['mean'].values
    if len(returns_by_decile) >= 2:
        mono_corr, _ = spearmanr(range(len(returns_by_decile)), returns_by_decile)
    else:
        mono_corr = 0
    fig.add_annotation(
        x=0.5, y=1.05, xref='x domain', yref='y domain',
        text=f'Monotonicity (Spearman): {mono_corr:.3f} | Spread D10-D1: {(returns_by_decile[-1]-returns_by_decile[0])*100:.2f}%',
        showarrow=False, font=dict(size=12, color='#FFD700'),
        row=1, col=1,
    )

    # ═══ 2. Feature Importance (Top 25) ═══
    if importance is not None and len(importance) > 0:
        top_imp = importance.head(25)[::-1]
        imp_colors = []
        for f in top_imp.index:
            if 'poly' in f or 'pullback' in f:
                imp_colors.append('#FF5722')
            elif 'mom' in f or 'ret_' in f:
                imp_colors.append('#2196F3')
            elif any(x in f for x in ['amihud', 'spread', 'turnover', 'vol_ratio']):
                imp_colors.append('#FF9800')
            elif any(x in f for x in ['rsi', 'macd', 'bb_', 'stoch', 'cci', 'donchian', 'adx', 'ma_ratio']):
                imp_colors.append('#9C27B0')
            elif 'rvol' in f or 'vol_of' in f or 'skew' in f or 'kurt' in f:
                imp_colors.append('#E91E63')
            else:
                imp_colors.append('#4CAF50')
        fig.add_trace(go.Bar(
            y=top_imp.index, x=top_imp.values, orientation='h',
            marker_color=imp_colors, name='Importance',
            hovertemplate='%{y}: %{x:.4f}<extra></extra>',
        ), row=1, col=2)

    # ═══ 3. Rank IC over time ═══
    ic = stats_df['rank_ic']
    ic_rolling = ic.rolling(6, min_periods=1).mean()
    fig.add_trace(go.Bar(
        x=ic.index, y=ic,
        marker_color=['#4CAF50' if v > 0 else '#f44336' for v in ic],
        opacity=0.4, name='Rank IC',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ic_rolling.index, y=ic_rolling,
        line=dict(color='#FFD700', width=2.5), name='IC (6-period MA)',
    ), row=2, col=1)
    fig.add_hline(y=0, line_color='gray', row=2, col=1)
    fig.add_annotation(
        x=0.5, y=1.05, xref='x3 domain', yref='y3 domain',
        text=f'Avg IC: {ic.mean():.4f} | IC>0: {(ic > 0).mean()*100:.0f}% | IR: {ic.mean()/(ic.std()+1e-10):.2f}',
        showarrow=False, font=dict(size=12, color='#FFD700'),
        row=2, col=1,
    )

    # ═══ 4. IC distribution ═══
    fig.add_trace(go.Histogram(
        x=ic, nbinsx=30,
        marker_color='#2196F3', opacity=0.7, name='IC Dist',
    ), row=2, col=2)
    fig.add_vline(x=0, line_color='red', line_dash='dot', row=2, col=2)
    fig.add_vline(x=ic.mean(), line_color='#FFD700', line_dash='solid', row=2, col=2)

    # ═══ 5. Heatmap: decile returns per period ═══
    pivot = records_df.groupby(['date', 'decile'])['fwd_return'].mean().unstack(fill_value=0) * 100
    fig.add_trace(go.Heatmap(
        z=pivot.values.T,
        x=[d.strftime('%Y-%m') for d in pivot.index],
        y=[f'D{d}' for d in pivot.columns],
        colorscale='RdYlGn', zmid=0,
        colorbar=dict(title='Return %', len=0.2, y=0.5),
        hovertemplate='%{x} D%{y}: %{z:.1f}%<extra></extra>',
    ), row=3, col=1)

    # ═══ 6. Cumulative L/S spread ═══
    cum_spread = stats_df['spread'].cumsum() * 100
    fig.add_trace(go.Scatter(
        x=cum_spread.index, y=cum_spread,
        fill='tozeroy',
        line=dict(color='#03A9F4', width=2),
        name='Cumulative Spread',
    ), row=3, col=2)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=3, col=2)

    # ═══ 7. Top vs Bottom decile over time ═══
    fig.add_trace(go.Scatter(
        x=stats_df.index, y=stats_df['top_decile_ret'] * 100,
        name=f'Top Decile (D{N_DECILES})',
        line=dict(color='#4CAF50', width=2),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=stats_df.index, y=stats_df['bot_decile_ret'] * 100,
        name='Bottom Decile (D1)',
        line=dict(color='#f44336', width=2),
    ), row=4, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=4, col=1)

    # ═══ 8. Box plot per decile ═══
    for d in sorted(records_df['decile'].unique()):
        subset = records_df[records_df['decile'] == d]['fwd_return'] * 100
        fig.add_trace(go.Box(
            y=subset, name=f'D{d}',
            marker_color=colors[int(d) - 1] if int(d) <= len(colors) else '#888',
            boxmean='sd',
            showlegend=False,
        ), row=4, col=2)

    # ═══ 9. Hit rate by decile ═══
    hit_rate = records_df.groupby('decile')['fwd_return'].apply(
        lambda x: (x > 0).mean() * 100
    )
    fig.add_trace(go.Bar(
        x=[f'D{d}' for d in hit_rate.index],
        y=hit_rate.values,
        marker_color=colors[:len(hit_rate)],
        text=[f'{v:.0f}%' for v in hit_rate],
        textposition='outside',
        name='Hit Rate %',
        showlegend=False,
    ), row=5, col=1)
    fig.add_hline(y=50, line_color='gray', line_dash='dot', row=5, col=1)

    # ═══ 10. Decile Sharpe approximation ═══
    decile_sharpe = records_df.groupby('decile')['fwd_return'].apply(
        lambda x: x.mean() / (x.std() + 1e-10)
    )
    fig.add_trace(go.Bar(
        x=[f'D{d}' for d in decile_sharpe.index],
        y=decile_sharpe.values,
        marker_color=colors[:len(decile_sharpe)],
        text=[f'{v:.3f}' for v in decile_sharpe],
        textposition='outside',
        name='Sharpe (approx)',
        showlegend=False,
    ), row=5, col=2)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=5, col=2)

    # ═══ Summary ═══
    avg_spread = stats_df['spread'].mean() * 100
    avg_ic = ic.mean()
    total_periods = len(stats_df)

    fig.update_layout(
        height=2600, width=1400,
        template='plotly_dark',
        title_text=(
            f'Cross-Factor Decile Calibration — {model_type.upper()}<br>'
            f'<sub>Avg IC: {avg_ic:.4f} | Spread D{N_DECILES}-D1: '
            f'{avg_spread:.2f}%/period | Monotonicity: {mono_corr:.3f} | '
            f'{total_periods} periods ({HOLDING_DAYS}d holding)</sub>'
        ),
        showlegend=True,
        legend=dict(orientation='h', y=-0.02),
    )

    path = os.path.join(RESULTS_DIR, f'decile_calibration_{model_type}.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard saved: {path}")
    return path


def print_decile_table(records_df):
    """Print clean decile table."""
    stats = records_df.groupby('decile').agg(
        mean_ret=('fwd_return', 'mean'),
        median_ret=('fwd_return', 'median'),
        std_ret=('fwd_return', 'std'),
        hit_rate=('fwd_return', lambda x: (x > 0).mean()),
        n_obs=('fwd_return', 'count'),
    )
    stats['mean_ret_pct'] = stats['mean_ret'] * 100
    stats['median_ret_pct'] = stats['median_ret'] * 100
    stats['std_ret_pct'] = stats['std_ret'] * 100
    stats['hit_rate_pct'] = stats['hit_rate'] * 100
    stats['sharpe_approx'] = stats['mean_ret'] / (stats['std_ret'] + 1e-10)

    print(f"\n{'='*80}")
    print(f"  DECILE RETURN TABLE ({HOLDING_DAYS}-day forward return)")
    print(f"{'='*80}")
    print(f"{'Decile':>8} {'Mean%':>8} {'Median%':>9} {'Std%':>8} {'Hit%':>7} {'Sharpe':>8} {'N':>7}")
    print(f"{'-'*80}")
    for d, row in stats.iterrows():
        print(f"  D{d:<5} {row['mean_ret_pct']:>8.2f} {row['median_ret_pct']:>9.2f} "
              f"{row['std_ret_pct']:>8.2f} {row['hit_rate_pct']:>7.1f} "
              f"{row['sharpe_approx']:>8.3f} {int(row['n_obs']):>7}")
    print(f"{'-'*80}")

    # Spread
    d1 = stats.loc[1, 'mean_ret'] if 1 in stats.index else stats.iloc[0]['mean_ret']
    d10 = stats.loc[N_DECILES, 'mean_ret'] if N_DECILES in stats.index else stats.iloc[-1]['mean_ret']
    print(f"  D{N_DECILES}-D1 Spread: {(d10 - d1)*100:.2f}%")

    # Monotonicity
    rets = stats['mean_ret'].values
    mono, _ = spearmanr(range(len(rets)), rets)
    print(f"  Monotonicity (Spearman): {mono:.3f}")
    print(f"{'='*80}")

    return stats


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

def main():
    print("Loading and computing features...")
    panel = load_panel()

    feat_cols = get_feature_cols(panel)
    print(f"Features available: {len(feat_cols)}")

    # Run for both models
    results = {}
    for mtype in ['lgbm', 'catboost']:
        print(f"\n{'#'*60}")
        print(f"  Running {mtype.upper()} decile analysis...")
        print(f"{'#'*60}")

        records_df, stats_df, importance = run_decile_analysis(panel, feat_cols, model_type=mtype)

        if records_df.empty:
            print(f"  {mtype}: No results!")
            continue

        print_decile_table(records_df)

        # Save raw data
        records_df.to_parquet(os.path.join(RESULTS_DIR, f'decile_records_{mtype}.parquet'))
        stats_df.to_parquet(os.path.join(RESULTS_DIR, f'decile_stats_{mtype}.parquet'))
        if len(importance) > 0:
            importance.to_json(os.path.join(RESULTS_DIR, f'decile_importance_{mtype}.json'))

        # Build dashboard
        path = build_calibration_dashboard(records_df, stats_df, mtype, importance)
        results[mtype] = (records_df, stats_df, importance, path)

    # Open dashboards
    import webbrowser
    for mtype, (*_, path) in results.items():
        webbrowser.open(f'file:///{os.path.abspath(path)}')

    return results


if __name__ == '__main__':
    results = main()
