"""
Failure Analysis — What explains when the model works vs fails?
================================================================
Uses the period-by-period diagnostics to find variables that
predict model success/failure.

Approach:
1. Compute market context variables per period
2. Regress IC ~ context variables (OLS, Lasso)
3. PCA on context to find latent failure modes
4. Decision tree to find simple rules
5. Visualize the "failure regime"
"""
import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRanker
from scipy.stats import spearmanr, rankdata, pearsonr
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _spearman_corr,
    _sample_pairs_and_train_epoch, _train_model, _predict,
    LR, L1_REG, L2_REG, MIN_COINS, VOL_FLOOR_PCT,
    N_EPOCHS, PAIRS_PER_DATE, NEAR_TIE_PCT, TAIL_WEIGHT_POW,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

TRAIN_DAYS = 540
PURGE_DAYS = 14
TRAIN_LABEL = 7
HOLD_EVAL = 14
REBAL_DAYS = 7
N_DECILES = 10

CATBOOST_PARAMS = {
    'loss_function': 'YetiRank', 'iterations': 200, 'depth': 4,
    'learning_rate': 0.05, 'l2_leaf_reg': 5.0, 'random_strength': 2.0,
    'bagging_temperature': 1.0, 'border_count': 64, 'verbose': 0,
    'random_seed': 42, 'task_type': 'CPU',
}


def main():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    n_feat = len([f for f in feat_cols if f in panel.columns])

    for h in [1, 7, 14, 28, 56, 90]:
        col = f'fwd_ret_{h}d'
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(h).shift(-h))

    # BTC data for regime variables
    btc = None
    if 'BTCUSDT' in panel.index.get_level_values('symbol').unique():
        btc = panel.xs('BTCUSDT', level='symbol')[['close', 'high', 'low', 'volume']].copy()
        btc['ret_7d'] = btc['close'].pct_change(7)
        btc['ret_14d'] = btc['close'].pct_change(14)
        btc['ret_28d'] = btc['close'].pct_change(28)
        btc['ret_56d'] = btc['close'].pct_change(56)
        btc['ret_90d'] = btc['close'].pct_change(90)
        btc['vol_14d'] = btc['close'].pct_change(1).rolling(14).std() * np.sqrt(365)
        btc['vol_28d'] = btc['close'].pct_change(1).rolling(28).std() * np.sqrt(365)
        btc['vol_56d'] = btc['close'].pct_change(1).rolling(56).std() * np.sqrt(365)
        btc['sma_50'] = btc['close'].rolling(50).mean()
        btc['sma_200'] = btc['close'].rolling(200).mean()
        btc['above_sma50'] = (btc['close'] > btc['sma_50']).astype(float)
        btc['above_sma200'] = (btc['close'] > btc['sma_200']).astype(float)
        btc['vol_change'] = btc['vol_14d'] / (btc['vol_56d'] + 1e-10)
        btc['trend_strength'] = btc['ret_28d'] / (btc['vol_28d'] + 1e-10)

    # Build date groups
    print("Building date groups with rich context...")
    date_groups = {}
    for date, group in panel.groupby(level='date'):
        g = group.dropna(subset=['close'])
        if len(g) < MIN_COINS: continue
        if 'turnover_28d' in g.columns:
            thresh = g['turnover_28d'].quantile(VOL_FLOOR_PCT)
            g = g[g['turnover_28d'] >= thresh]
        if len(g) < MIN_COINS: continue

        available = [f for f in feat_cols if f in g.columns]
        features = g[available].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = _cross_sectional_rank_normalize(features)

        train_ret = np.nan_to_num(g[f'fwd_ret_{TRAIN_LABEL}d'].values.astype(np.float64), nan=0.0)
        eval_ret = np.nan_to_num(g[f'fwd_ret_{HOLD_EVAL}d'].values.astype(np.float64), nan=0.0)
        train_excess = train_ret - np.nanmean(train_ret)

        # Rich context variables
        ctx = {}
        # Cross-sectional
        ctx['n_coins'] = len(g)
        ctx['xs_dispersion'] = np.std(eval_ret) if len(eval_ret) > 1 else 0
        ctx['xs_mean'] = np.mean(eval_ret)
        ctx['xs_skew'] = pd.Series(eval_ret).skew() if len(eval_ret) > 3 else 0
        ctx['xs_kurt'] = pd.Series(eval_ret).kurtosis() if len(eval_ret) > 4 else 0
        ctx['breadth'] = np.mean(eval_ret > 0)
        ctx['pct_extreme_up'] = np.mean(eval_ret > 0.3)
        ctx['pct_extreme_down'] = np.mean(eval_ret < -0.3)
        ctx['winner_loser_ratio'] = (np.mean(eval_ret[eval_ret > 0]) / (-np.mean(eval_ret[eval_ret < 0]) + 1e-10)) if np.any(eval_ret < 0) and np.any(eval_ret > 0) else 1.0
        ctx['top10_vs_bot10'] = np.mean(np.sort(eval_ret)[-max(1,len(eval_ret)//10):]) - np.mean(np.sort(eval_ret)[:max(1,len(eval_ret)//10)])

        # Correlation structure
        if 'ret_7d' in g.columns:
            r7 = g['ret_7d'].dropna().values
            if len(r7) > 10:
                # Average pairwise correlation proxy: var(mean) / mean(var)
                ctx['avg_corr_proxy'] = np.var(r7) / (np.mean(r7**2) + 1e-10)

        # BTC context
        if btc is not None:
            for col in ['ret_7d', 'ret_14d', 'ret_28d', 'ret_56d', 'ret_90d',
                        'vol_14d', 'vol_28d', 'vol_change', 'above_sma50',
                        'above_sma200', 'trend_strength']:
                val = btc[col].get(date, np.nan) if col in btc.columns else np.nan
                ctx[f'btc_{col}'] = val if np.isfinite(val) else 0.0

        date_groups[date] = {
            'features': features, 'train_excess': train_excess,
            'train_ret': train_ret, 'eval_ret': eval_ret,
            'context': ctx,
        }

    sorted_dates = sorted(date_groups.keys())
    print(f"  {len(sorted_dates)} dates, {n_feat} features")

    # JIT warmup
    d, e, w = np.random.randn(20, n_feat), np.random.randn(20), np.zeros(n_feat)
    _sample_pairs_and_train_epoch(d, e, w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # ═══ Run CatBoost 540d (best model) and collect per-period data ═══
    print("\nRunning CatBoost 540d to collect period diagnostics...")
    start_idx = 0
    for i, dd in enumerate(sorted_dates):
        if (dd - sorted_dates[0]).days >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i; break
    pred_dates = sorted_dates[start_idx::REBAL_DAYS]

    records = []
    model = None
    t0 = time.time()

    for ri, pred_date in enumerate(pred_dates):
        if ri % 4 == 0 or model is None:
            te = pred_date - pd.Timedelta(days=PURGE_DAYS)
            ts = te - pd.Timedelta(days=TRAIN_DAYS)
            td = [(date_groups[d]['features'][np.abs(date_groups[d]['train_ret']) > 1e-10],
                   date_groups[d]['train_excess'][np.abs(date_groups[d]['train_ret']) > 1e-10])
                  for d in sorted_dates if ts <= d <= te and d in date_groups
                  and np.sum(np.abs(date_groups[d]['train_ret']) > 1e-10) >= 5]
            if len(td) < 10: continue

            all_X, all_y, gids = [], [], []
            gid = 0
            for feat, excess in td:
                if len(excess) < 5: continue
                try: labels = pd.qcut(excess, 5, labels=False, duplicates='drop')
                except: labels = np.clip((rankdata(excess)*5/(len(excess)+1)).astype(int), 0, 4)
                all_X.append(feat); all_y.append(labels); gids.extend([gid]*len(labels)); gid += 1
            if len(all_X) >= 5:
                model = CatBoostRanker(**CATBOOST_PARAMS)
                model.fit(np.nan_to_num(np.vstack(all_X), nan=0),
                          np.concatenate(all_y).astype(int), group_id=np.array(gids))

        if model is None or pred_date not in date_groups: continue
        dg = date_groups[pred_date]
        scores = model.predict(np.nan_to_num(dg['features'], nan=0.0))
        er = dg['eval_ret']
        excess = er - np.mean(er)
        ic = float(_spearman_corr(scores, excess))

        si = np.argsort(scores)
        n = len(scores)
        dm = np.zeros(N_DECILES); dc = np.zeros(N_DECILES)
        for rp, idx in enumerate(si):
            db = min(rp * N_DECILES // n, N_DECILES - 1)
            dm[db] += er[idx]; dc[db] += 1
        dm = np.where(dc > 0, dm / dc, 0.0)
        mono, _ = spearmanr(range(N_DECILES), dm)
        spread = (dm[-1] - dm[0]) * 100

        records.append({
            'date': pred_date, 'ic': ic, 'mono': mono, 'spread': spread,
            'd1_ret': dm[0]*100, 'd10_ret': dm[-1]*100,
            **dg['context'],
        })

        if (ri+1) % 30 == 0:
            print(f"  [{ri+1}/{len(pred_dates)}] {pred_date.date()} [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    print(f"  {len(df)} periods collected")

    # ═══ ANALYSIS 1: Correlation of IC with context variables ═══
    print("\n=== ANALYSIS 1: What correlates with IC? ===")
    context_cols = [c for c in df.columns if c not in
                    ['ic', 'mono', 'spread', 'd1_ret', 'd10_ret'] +
                    [f'dm_{i}' for i in range(10)]]

    corr_with_ic = {}
    for col in context_cols:
        vals = df[col].values
        valid = np.isfinite(vals) & np.isfinite(df['ic'].values)
        if valid.sum() > 20:
            r, p = pearsonr(vals[valid], df['ic'].values[valid])
            corr_with_ic[col] = {'r': r, 'p': p, 'abs_r': abs(r)}

    corr_df = pd.DataFrame(corr_with_ic).T.sort_values('abs_r', ascending=False)
    print(f"\n{'Variable':<30} {'Corr':>8} {'p-value':>10} {'Significant':>12}")
    print(f"{'-'*65}")
    for var, row in corr_df.head(20).iterrows():
        sig = '***' if row['p'] < 0.01 else '**' if row['p'] < 0.05 else '*' if row['p'] < 0.1 else ''
        print(f"{var:<30} {row['r']:>8.3f} {row['p']:>10.4f} {sig:>12}")

    # ═══ ANALYSIS 2: OLS regression IC ~ context ═══
    print("\n=== ANALYSIS 2: OLS Regression IC ~ context ===")
    X_cols = [c for c in context_cols if df[c].notna().mean() > 0.8]
    X = df[X_cols].fillna(0).values
    y = df['ic'].values
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_v, y_v = X[valid], y[valid]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_v)

    # OLS
    ols = LinearRegression()
    ols.fit(X_scaled, y_v)
    r2_ols = ols.score(X_scaled, y_v)
    print(f"  OLS R-squared: {r2_ols:.4f}")

    # Lasso (feature selection)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(X_scaled, y_v)
    r2_lasso = lasso.score(X_scaled, y_v)
    print(f"  Lasso R-squared: {r2_lasso:.4f} (alpha={lasso.alpha_:.4f})")

    lasso_coefs = pd.Series(lasso.coef_, index=X_cols).sort_values(key=abs, ascending=False)
    active = lasso_coefs[lasso_coefs.abs() > 1e-6]
    print(f"\n  Lasso selected {len(active)} variables:")
    for var, coef in active.head(15).items():
        print(f"    {var:<30} {coef:>8.4f}")

    # ═══ ANALYSIS 3: GBM to predict IC ═══
    print("\n=== ANALYSIS 3: GBM predicting IC (non-linear) ===")
    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=42)
    gbm.fit(X_scaled, y_v)
    r2_gbm = gbm.score(X_scaled, y_v)
    print(f"  GBM R-squared: {r2_gbm:.4f}")

    gbm_imp = pd.Series(gbm.feature_importances_, index=X_cols).sort_values(ascending=False)
    print(f"\n  GBM Top 15 features:")
    for var, imp in gbm_imp.head(15).items():
        print(f"    {var:<30} {imp:>8.4f}")

    # ═══ ANALYSIS 4: PCA on context ═══
    print("\n=== ANALYSIS 4: PCA on context variables ===")
    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Explained variance: {pca.explained_variance_ratio_[:5]}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_[:5])}")

    # Correlate PCA components with IC
    for pc in range(min(5, X_pca.shape[1])):
        r, p = pearsonr(X_pca[:, pc], y_v)
        print(f"  PC{pc+1} vs IC: r={r:.3f} (p={p:.4f})")
        if abs(r) > 0.1:
            # Show what this PC loads on
            loadings = pd.Series(pca.components_[pc], index=X_cols).sort_values(key=abs, ascending=False)
            print(f"    Top loadings: {dict(loadings.head(5).items())}")

    # ═══ ANALYSIS 5: Conditional analysis ═══
    print("\n=== ANALYSIS 5: Conditional IC analysis ===")

    conditions = {
        'BTC bull (ret_28d > 0)': df['btc_ret_28d'] > 0,
        'BTC bear (ret_28d < 0)': df['btc_ret_28d'] < 0,
        'BTC above SMA200': df['btc_above_sma200'] > 0.5,
        'BTC below SMA200': df['btc_above_sma200'] < 0.5,
        'High vol (vol_14d > median)': df['btc_vol_14d'] > df['btc_vol_14d'].median(),
        'Low vol': df['btc_vol_14d'] <= df['btc_vol_14d'].median(),
        'Vol expanding': df['btc_vol_change'] > 1.0,
        'Vol contracting': df['btc_vol_change'] <= 1.0,
        'High dispersion': df['xs_dispersion'] > df['xs_dispersion'].median(),
        'Low dispersion': df['xs_dispersion'] <= df['xs_dispersion'].median(),
        'Broad market up (breadth>60%)': df['breadth'] > 0.6,
        'Narrow/down (breadth<40%)': df['breadth'] < 0.4,
        'Many extremes (>5% coins +30%)': df['pct_extreme_up'] > 0.05,
        'Few extremes': df['pct_extreme_up'] <= 0.05,
        'Strong BTC trend': df['btc_trend_strength'].abs() > 1.0,
        'Weak BTC trend': df['btc_trend_strength'].abs() <= 1.0,
    }

    print(f"\n{'Condition':<40} {'IC':>7} {'IC>0':>6} {'Mono':>7} {'Spread':>8} {'N':>5}")
    print(f"{'-'*75}")
    cond_results = []
    for name, mask in conditions.items():
        mask = mask.fillna(False)
        sub = df[mask]
        if len(sub) < 10: continue
        avg_ic = sub['ic'].mean()
        ic_pos = (sub['ic'] > 0).mean() * 100
        avg_mono = sub['mono'].mean()
        avg_spread = sub['spread'].mean()
        print(f"{name:<40} {avg_ic:>7.4f} {ic_pos:>5.0f}% {avg_mono:>7.3f} {avg_spread:>7.2f}% {len(sub):>5}")
        cond_results.append({'name': name, 'ic': avg_ic, 'ic_pos': ic_pos,
                            'mono': avg_mono, 'spread': avg_spread, 'n': len(sub)})

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed/60:.1f} min")

    # ═══ Dashboard ═══
    build_dashboard(df, corr_df, lasso_coefs, gbm_imp, X_pca, y_v, X_cols, pca,
                    cond_results, conditions, elapsed)


def build_dashboard(df, corr_df, lasso_coefs, gbm_imp, X_pca, y_v, X_cols, pca,
                    cond_results, conditions, elapsed):

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Correlation: Context Variables vs IC',
            'Lasso Coefficients (IC predictors)',
            'IC vs BTC 28d Return', 'IC vs Cross-Sectional Dispersion',
            'IC vs BTC Volatility', 'IC by Regime (conditional)',
            'PCA: PC1 vs IC', 'GBM Feature Importance',
        ],
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    # 1. Correlation bars
    top_corr = corr_df.head(15).iloc[::-1]
    colors = ['#4CAF50' if r > 0 else '#f44336' for r in top_corr['r']]
    fig.add_trace(go.Bar(
        y=top_corr.index, x=top_corr['r'], orientation='h',
        marker_color=colors, showlegend=False,
    ), row=1, col=1)

    # 2. Lasso coefficients
    active = lasso_coefs[lasso_coefs.abs() > 1e-6].sort_values(key=abs, ascending=True).tail(15)
    colors_l = ['#4CAF50' if c > 0 else '#f44336' for c in active]
    fig.add_trace(go.Bar(
        y=active.index, x=active.values, orientation='h',
        marker_color=colors_l, showlegend=False,
    ), row=1, col=2)

    # 3. IC vs BTC return
    valid = df['btc_ret_28d'].notna()
    fig.add_trace(go.Scatter(
        x=df.loc[valid, 'btc_ret_28d'] * 100, y=df.loc[valid, 'ic'],
        mode='markers', marker=dict(
            color=df.loc[valid].index.year, colorscale='Viridis',
            size=5, opacity=0.6, colorbar=dict(title='Year', len=0.2, y=0.62)),
        showlegend=False,
    ), row=2, col=1)
    fig.update_xaxes(title_text='BTC 28d Return %', row=2, col=1)
    fig.update_yaxes(title_text='Rank IC', row=2, col=1)

    # 4. IC vs dispersion
    fig.add_trace(go.Scatter(
        x=df['xs_dispersion'] * 100, y=df['ic'],
        mode='markers', marker=dict(
            color=df.index.year, colorscale='Viridis',
            size=5, opacity=0.6),
        showlegend=False,
    ), row=2, col=2)
    fig.update_xaxes(title_text='XS Dispersion %', row=2, col=2)

    # 5. IC vs BTC vol
    valid_v = df['btc_vol_14d'].notna()
    fig.add_trace(go.Scatter(
        x=df.loc[valid_v, 'btc_vol_14d'] * 100, y=df.loc[valid_v, 'ic'],
        mode='markers', marker=dict(
            color=df.loc[valid_v].index.year, colorscale='Viridis',
            size=5, opacity=0.6),
        showlegend=False,
    ), row=3, col=1)
    fig.update_xaxes(title_text='BTC 14d Vol %', row=3, col=1)

    # 6. Conditional IC bars
    cr = pd.DataFrame(cond_results)
    if not cr.empty:
        colors_c = ['#4CAF50' if ic > 0.08 else '#FF9800' if ic > 0.04 else '#f44336'
                     for ic in cr['ic']]
        fig.add_trace(go.Bar(
            y=cr['name'], x=cr['ic'], orientation='h',
            marker_color=colors_c, showlegend=False,
            text=[f"{ic:.3f}" for ic in cr['ic']], textposition='outside',
        ), row=3, col=2)
        fig.add_vline(x=0, line_color='gray', row=3, col=2)

    # 7. PCA PC1 vs IC
    if X_pca is not None and len(y_v) > 0:
        fig.add_trace(go.Scatter(
            x=X_pca[:, 0], y=y_v, mode='markers',
            marker=dict(size=5, opacity=0.5, color='#03A9F4'),
            showlegend=False,
        ), row=4, col=1)
        fig.update_xaxes(title_text='PC1', row=4, col=1)
        fig.update_yaxes(title_text='IC', row=4, col=1)

    # 8. GBM importance
    top_gbm = gbm_imp.head(15).iloc[::-1]
    fig.add_trace(go.Bar(
        y=top_gbm.index, x=top_gbm.values, orientation='h',
        marker_color='#FF9800', showlegend=False,
    ), row=4, col=2)

    fig.update_layout(
        height=2200, width=1500, template='plotly_dark',
        title_text=f'Failure Analysis: What Explains When the Model Works?<br>'
                   f'<sub>OLS R2={0:.3f} | Lasso R2={0:.3f} | GBM R2={0:.3f} | {elapsed/60:.1f}min</sub>',
        showlegend=False,
    )

    path = os.path.join(RESULTS_DIR, 'failure_analysis.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"Dashboard: {path}")
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
