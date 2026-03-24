"""
Model Diagnostic — When does it work? When does it fail?
==========================================================
Goes back to the calibration framework (best signal).
Analyzes IC, monotonicity, spread BY YEAR, BY REGIME, BY CONDITION.

Questions to answer:
1. Which model is most homogeneous across time?
2. In which year does it fail?
3. WHEN does it fail? (bull/bear/sideways, high/low vol, etc.)
"""
import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRanker
from scipy.stats import spearmanr, rankdata
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

N_DECILES = 10
REBAL_DAYS = 7  # weekly predictions for finer granularity
PURGE_DAYS = 14
TRAIN_LABEL = 7
HOLD_EVAL = 14  # evaluate at 14d

CATBOOST_PARAMS = {
    'loss_function': 'YetiRank', 'iterations': 200, 'depth': 4,
    'learning_rate': 0.05, 'l2_leaf_reg': 5.0, 'random_strength': 2.0,
    'bagging_temperature': 1.0, 'border_count': 64, 'verbose': 0,
    'random_seed': 42, 'task_type': 'CPU',
}

# Two models to compare
MODELS = {
    'catboost_540': {'type': 'catboost', 'train_days': 540, 'label': 'CatBoost 540d'},
    'catboost_180': {'type': 'catboost', 'train_days': 180, 'label': 'CatBoost 180d'},
    'linear_540':   {'type': 'linear',   'train_days': 540, 'label': 'Linear 540d'},
    'linear_180':   {'type': 'linear',   'train_days': 180, 'label': 'Linear 180d'},
}


def train_catboost(train_data):
    all_X, all_y, gids = [], [], []
    gid = 0
    for feat, excess in train_data:
        if len(excess) < 5: continue
        try:
            labels = pd.qcut(excess, 5, labels=False, duplicates='drop')
        except ValueError:
            labels = np.clip((rankdata(excess) * 5 / (len(excess)+1)).astype(int), 0, 4)
        all_X.append(feat); all_y.append(labels); gids.extend([gid]*len(labels)); gid += 1
    if len(all_X) < 5: return None
    m = CatBoostRanker(**CATBOOST_PARAMS)
    m.fit(np.nan_to_num(np.vstack(all_X), nan=0), np.concatenate(all_y).astype(int), group_id=np.array(gids))
    return m


def train_linear(train_data, n_feat):
    tf, te, offsets = [], [], [0]
    for feat, excess in train_data:
        if len(excess) < 5: continue
        tf.append(feat); te.append(excess); offsets.append(offsets[-1]+len(excess))
    if len(tf) < 5: return None
    return _train_model(np.vstack(tf), np.concatenate(te),
        np.array(offsets, dtype=np.int64), n_feat, N_EPOCHS, PAIRS_PER_DATE,
        NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG, 42)


def main():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    n_feat = len([f for f in feat_cols if f in panel.columns])

    # Ensure columns
    for h in [TRAIN_LABEL, HOLD_EVAL]:
        col = f'fwd_ret_{h}d'
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(h).shift(-h))

    # BTC returns for regime detection
    btc_data = {}
    if 'BTCUSDT' in panel.index.get_level_values('symbol').unique():
        btc = panel.xs('BTCUSDT', level='symbol')
        for h in [7, 14, 28, 56]:
            btc_data[f'btc_ret_{h}d'] = btc['close'].pct_change(h)
        btc_data['btc_vol_28d'] = btc['close'].pct_change(1).rolling(28).std() * np.sqrt(365)

    # Build date groups
    print("Building date groups...")
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

        # Market context
        n_coins = len(g)
        xs_dispersion = np.std(eval_ret) if len(eval_ret) > 1 else 0
        mkt_ret = np.mean(eval_ret)
        breadth = np.mean(eval_ret > 0)

        btc_r14 = btc_data.get('btc_ret_14d', pd.Series(dtype=float)).get(date, 0)
        btc_r56 = btc_data.get('btc_ret_56d', pd.Series(dtype=float)).get(date, 0)
        btc_vol = btc_data.get('btc_vol_28d', pd.Series(dtype=float)).get(date, 0)

        date_groups[date] = {
            'features': features, 'train_excess': train_excess,
            'train_ret': train_ret, 'eval_ret': eval_ret,
            'n_coins': n_coins,
            'context': {
                'mkt_ret': mkt_ret, 'dispersion': xs_dispersion,
                'breadth': breadth, 'btc_ret_14d': btc_r14,
                'btc_ret_56d': btc_r56, 'btc_vol_28d': btc_vol,
            }
        }

    sorted_dates = sorted(date_groups.keys())
    print(f"  {len(sorted_dates)} dates, {n_feat} features")

    # JIT warmup
    d, e, w = np.random.randn(20, n_feat), np.random.randn(20), np.zeros(n_feat)
    _sample_pairs_and_train_epoch(d, e, w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # ═══ Run each model ═══
    all_diagnostics = {}

    for mname, mcfg in MODELS.items():
        print(f"\n  Running {mcfg['label']}...")
        train_days = mcfg['train_days']
        start_idx = 0
        for i, dd in enumerate(sorted_dates):
            if (dd - sorted_dates[0]).days >= train_days + PURGE_DAYS:
                start_idx = i; break
        pred_dates = sorted_dates[start_idx::REBAL_DAYS]

        records = []
        model = None
        retrain_every = 4
        t0 = time.time()

        for ri, pred_date in enumerate(pred_dates):
            if ri % retrain_every == 0 or model is None:
                te = pred_date - pd.Timedelta(days=PURGE_DAYS)
                ts = te - pd.Timedelta(days=train_days)
                td = [(date_groups[d]['features'][np.abs(date_groups[d]['train_ret']) > 1e-10],
                       date_groups[d]['train_excess'][np.abs(date_groups[d]['train_ret']) > 1e-10])
                      for d in sorted_dates if ts <= d <= te and d in date_groups
                      and np.sum(np.abs(date_groups[d]['train_ret']) > 1e-10) >= 5]
                if len(td) < 10: continue
                if mcfg['type'] == 'catboost':
                    model = train_catboost(td)
                else:
                    model = train_linear(td, n_feat)

            if model is None or pred_date not in date_groups: continue

            dg = date_groups[pred_date]
            X = np.nan_to_num(dg['features'], nan=0.0)
            if mcfg['type'] == 'catboost':
                scores = model.predict(X)
            else:
                scores = X @ model

            n = dg['n_coins']
            er = dg['eval_ret']
            excess = er - np.mean(er)

            # IC
            ic = float(_spearman_corr(scores, excess))

            # Decile means
            si = np.argsort(scores)
            dm = np.zeros(N_DECILES)
            dc = np.zeros(N_DECILES)
            for rp, idx in enumerate(si):
                db = min(rp * N_DECILES // n, N_DECILES - 1)
                dm[db] += er[idx]; dc[db] += 1
            dm = np.where(dc > 0, dm / dc, 0.0)
            mono, _ = spearmanr(range(N_DECILES), dm)
            spread = dm[N_DECILES-1] - dm[0]

            # Top/bottom returns
            n10 = max(1, n // 10)
            d1_ret = np.mean(er[si[:n10]])
            d10_ret = np.mean(er[si[-n10:]])

            records.append({
                'date': pred_date,
                'year': pred_date.year,
                'month': pred_date.month,
                'ic': ic,
                'mono': mono,
                'spread': spread * 100,
                'd1_ret': d1_ret * 100,
                'd10_ret': d10_ret * 100,
                'n_coins': n,
                **{f'dm_{i+1}': dm[i]*100 for i in range(N_DECILES)},
                **{k: v for k, v in dg['context'].items()},
            })

        df = pd.DataFrame(records)
        all_diagnostics[mname] = df
        elapsed = time.time() - t0

        if len(df) > 0:
            print(f"    {len(df)} periods [{elapsed:.0f}s]")
            print(f"    Overall: IC={df['ic'].mean():.4f} Mono={df['mono'].mean():.3f} "
                  f"Spread={df['spread'].mean():.2f}%")

            # Per year
            print(f"    {'Year':>6} {'IC':>7} {'IC>0':>6} {'Mono':>6} {'Spread':>8} {'D1':>7} {'D10':>7} {'N':>5}")
            print(f"    {'-'*60}")
            for yr in sorted(df['year'].unique()):
                sub = df[df['year'] == yr]
                print(f"    {yr:>6} {sub['ic'].mean():>7.4f} {(sub['ic']>0).mean()*100:>5.0f}% "
                      f"{sub['mono'].mean():>6.3f} {sub['spread'].mean():>7.2f}% "
                      f"{sub['d1_ret'].mean():>7.2f} {sub['d10_ret'].mean():>7.2f} {len(sub):>5}")

    # ═══ Dashboard ═══
    print("\nBuilding diagnostic dashboard...")
    build_dashboard(all_diagnostics)
    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


def build_dashboard(all_diag):
    model_colors = {
        'catboost_540': '#4CAF50', 'catboost_180': '#FF9800',
        'linear_540': '#2196F3', 'linear_180': '#9C27B0',
    }

    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            'Rank IC Over Time (all models)', 'IC by Year (all models)',
            'Monotonicity Over Time', 'Monotonicity by Year',
            'D10-D1 Spread Over Time', 'Spread by Year',
            'IC vs BTC Return (regime)', 'IC vs Market Dispersion',
            'Decile Profile by Year (best model)', 'Model Consistency (IC std by year)',
        ],
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
    )

    # ═══ Row 1: IC over time + by year ═══
    for mname, df in all_diag.items():
        if df.empty: continue
        ic_roll = df['ic'].rolling(8, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['date'], y=ic_roll, name=MODELS[mname]['label'],
            line=dict(color=model_colors[mname], width=2),
        ), row=1, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=1, col=1)

    # IC by year bars
    years = sorted(set(yr for df in all_diag.values() for yr in df['year'].unique()))
    for mname, df in all_diag.items():
        if df.empty: continue
        yearly_ic = [df[df['year']==yr]['ic'].mean() for yr in years]
        fig.add_trace(go.Bar(
            x=[str(y) for y in years], y=yearly_ic, name=MODELS[mname]['label'],
            marker_color=model_colors[mname], opacity=0.7, showlegend=False,
        ), row=1, col=2)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=1, col=2)

    # ═══ Row 2: Monotonicity ═══
    for mname, df in all_diag.items():
        if df.empty: continue
        mono_roll = df['mono'].rolling(8, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['date'], y=mono_roll, name=None,
            line=dict(color=model_colors[mname], width=2), showlegend=False,
        ), row=2, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=2, col=1)

    for mname, df in all_diag.items():
        if df.empty: continue
        yearly_mono = [df[df['year']==yr]['mono'].mean() for yr in years]
        fig.add_trace(go.Bar(
            x=[str(y) for y in years], y=yearly_mono,
            marker_color=model_colors[mname], opacity=0.7, showlegend=False,
        ), row=2, col=2)

    # ═══ Row 3: Spread ═══
    for mname, df in all_diag.items():
        if df.empty: continue
        sp_roll = df['spread'].rolling(8, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['date'], y=sp_roll, name=None,
            line=dict(color=model_colors[mname], width=2), showlegend=False,
        ), row=3, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=3, col=1)

    for mname, df in all_diag.items():
        if df.empty: continue
        yearly_sp = [df[df['year']==yr]['spread'].mean() for yr in years]
        fig.add_trace(go.Bar(
            x=[str(y) for y in years], y=yearly_sp,
            marker_color=model_colors[mname], opacity=0.7, showlegend=False,
        ), row=3, col=2)

    # ═══ Row 4: IC vs regime ═══
    # Use the best model (catboost_540) for scatter
    best_name = 'catboost_540'
    if best_name in all_diag and not all_diag[best_name].empty:
        df = all_diag[best_name]

        # IC vs BTC return
        btc_r = df['btc_ret_14d'] * 100
        valid = np.isfinite(btc_r) & np.isfinite(df['ic'])
        if valid.sum() > 10:
            fig.add_trace(go.Scatter(
                x=btc_r[valid], y=df['ic'][valid], mode='markers',
                marker=dict(color=df['year'][valid], colorscale='Viridis', size=5, opacity=0.5),
                name='IC vs BTC 14d', showlegend=False,
            ), row=4, col=1)
            fig.update_xaxes(title_text='BTC 14d Return %', row=4, col=1)

        # IC vs dispersion
        disp = df['dispersion'] * 100
        valid2 = np.isfinite(disp) & np.isfinite(df['ic'])
        if valid2.sum() > 10:
            fig.add_trace(go.Scatter(
                x=disp[valid2], y=df['ic'][valid2], mode='markers',
                marker=dict(color=df['year'][valid2], colorscale='Viridis', size=5, opacity=0.5),
                name='IC vs Dispersion', showlegend=False,
            ), row=4, col=2)
            fig.update_xaxes(title_text='Cross-Sectional Dispersion %', row=4, col=2)

    # ═══ Row 5: Decile profile by year + consistency ═══
    if best_name in all_diag and not all_diag[best_name].empty:
        df = all_diag[best_name]
        yr_colors = {2021: '#f44336', 2022: '#FF9800', 2023: '#FFEB3B',
                     2024: '#4CAF50', 2025: '#2196F3', 2026: '#9C27B0'}

        for yr in sorted(df['year'].unique()):
            sub = df[df['year'] == yr]
            dm_cols = [f'dm_{i+1}' for i in range(N_DECILES)]
            dm_means = [sub[c].mean() for c in dm_cols]
            fig.add_trace(go.Scatter(
                x=[f'D{i+1}' for i in range(N_DECILES)], y=dm_means,
                name=str(yr), mode='lines+markers',
                line=dict(color=yr_colors.get(yr, '#888'), width=2),
                marker=dict(size=6),
            ), row=5, col=1)
        fig.add_hline(y=0, line_color='gray', line_dash='dot', row=5, col=1)

        # Consistency: IC std by year per model
        for mname, mdf in all_diag.items():
            if mdf.empty: continue
            yearly_std = [mdf[mdf['year']==yr]['ic'].std() for yr in years]
            fig.add_trace(go.Bar(
                x=[str(y) for y in years], y=yearly_std,
                marker_color=model_colors[mname], opacity=0.7, showlegend=False,
            ), row=5, col=2)

    # Summary
    summaries = []
    for mname, df in all_diag.items():
        if df.empty: continue
        ic = df['ic'].mean()
        ic_std = df.groupby('year')['ic'].mean().std()
        summaries.append(f"{MODELS[mname]['label']}: IC={ic:.3f} yearly_std={ic_std:.3f}")

    fig.update_layout(
        height=2400, width=1500, template='plotly_dark', barmode='group',
        title_text=f'Model Diagnostic: When Does It Work?<br><sub>{" | ".join(summaries)}</sub>',
        showlegend=True, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'model_diagnostic.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"Dashboard: {path}")
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
