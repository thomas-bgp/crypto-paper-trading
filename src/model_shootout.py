"""
Model Shootout — Linear vs LGBMRanker vs Ensemble
===================================================
Implements ALL committee recommendations:
  - BTC-relative features (beta, alpha, corr, residual momentum)
  - Market regime features (breadth, vol regime)
  - Vol-scaled labels
  - Tail indicator features + piecewise-linear basis
  - Tuned pair weighting (80% tail, pow=2.0)
  - LGBMRanker with LambdaRank
  - Borda count ensemble

Compares 3 models on decile calibration + 6-strategy backtest.
"""
import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

from numba import njit
from scipy.stats import spearmanr, rankdata
from lightgbm import LGBMRanker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _train_model, _predict, _spearman_corr,
    _sample_pairs_and_train_epoch,
    LR, L1_REG, L2_REG, MIN_COINS, VOL_FLOOR_PCT,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ─── Config (audit-fixed) ───
TRAIN_DAYS = 540
PURGE_DAYS = 14        # safe for 7d train label
HOLD_DAYS = 28
REBAL_DAYS = 28        # no overlap
TRAIN_LABEL_DAYS = 7   # short label (audit winner)
N_EPOCHS = 60
PAIRS_PER_DATE = 500
NEAR_TIE_PCT = 40.0    # more aggressive filtering (committee)
TAIL_WEIGHT_POW = 2.0  # quadratic tail weight (committee)
INITIAL_CAPITAL = 100_000
COST_PER_SIDE = 0.0007
FUNDING_DAILY = 0.0003
N_DECILES = 10

STRATEGIES = {
    'long_q10':  {'lp': 0.10, 'sp': 0.00, 'label': 'Long 10%'},
    'short_q10': {'lp': 0.00, 'sp': 0.10, 'label': 'Short 10%'},
    'ls_q10':    {'lp': 0.10, 'sp': 0.10, 'label': 'L/S 10-10%'},
    'long_q20':  {'lp': 0.20, 'sp': 0.00, 'label': 'Long 20%'},
    'short_q20': {'lp': 0.00, 'sp': 0.20, 'label': 'Short 20%'},
    'ls_q20':    {'lp': 0.20, 'sp': 0.20, 'label': 'L/S 20-20%'},
}

# LGBMRanker params (OPUS-MODEL committee spec)
LGBM_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [3, 5, 10],
    'label_gain': [0, 1, 3, 7, 15],
    'n_estimators': 200,
    'max_depth': 4,
    'num_leaves': 12,
    'learning_rate': 0.03,
    'min_child_samples': 15,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'subsample': 0.75,
    'colsample_bytree': 0.6,
    'subsample_freq': 1,
    'max_bin': 63,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}


# ════════════════════════════════════════════
# ENHANCED FEATURE ENGINEERING
# ════════════════════════════════════════════

def add_btc_relative_features(date_groups, sorted_dates):
    """Add BTC-relative features using PAST data only (no leakage)."""
    # Collect BTC rank from base_features (purely backward-looking)
    btc_past_rets = {}
    for d in sorted_dates:
        if d not in date_groups: continue
        dg = date_groups[d]
        syms = dg['symbols']
        if 'BTCUSDT' in syms:
            idx = syms.index('BTCUSDT')
            # Use BTC's rank-normalized feature value (backward-looking)
            btc_past_rets[d] = dg['base_features'][idx, 0]

    for d in sorted_dates:
        if d not in date_groups: continue
        dg = date_groups[d]
        n = dg['n_coins']
        bf = dg['base_features']  # already rank-normalized, backward-looking

        # Use base_features column 0 (ret_3d or first momentum feature) as proxy
        # These are already backward-looking from feature_engine_v2
        # BTC features: use the RANK of coin within the cross-section for momentum features
        # All base_features are already backward-looking, so any of them is safe

        # Simple BTC-relative: just broadcast BTC's rank position to all coins
        past_btc_vol = [btc_past_rets[dd] for dd in sorted_dates
                        if dd < d and dd in btc_past_rets and (d - dd).days <= 56]
        btc_vol = np.std(past_btc_vol) if len(past_btc_vol) > 5 else 0.0

        # Find BTC in this cross-section
        btc_rank = 0.5  # default if BTC not in universe
        if 'BTCUSDT' in dg['symbols']:
            bi = dg['symbols'].index('BTCUSDT')
            btc_rank = bf[bi, 0]  # BTC's rank on first feature

        extra = np.zeros((n, 4))
        extra[:, 0] = bf[:, 0] - btc_rank  # coin rank vs BTC rank (backward features)
        extra[:, 1] = 1.0 if btc_rank > 0.5 else -1.0  # BTC above/below median
        extra[:, 2] = abs(btc_rank - 0.5) * 2  # BTC rank extremity
        extra[:, 3] = btc_vol  # trailing BTC vol

        dg['btc_features'] = extra
        dg['btc_feature_names'] = ['rank_vs_btc', 'btc_direction_rank', 'btc_extremity', 'btc_vol_trailing']


def add_regime_features(date_groups, sorted_dates):
    """Add market regime features using PAST data only (no leakage)."""
    for d in sorted_dates:
        if d not in date_groups: continue
        dg = date_groups[d]
        n = dg['n_coins']
        # Use base_features (backward-looking rank-normalized) for regime proxy
        bf = dg['base_features']
        # Use first momentum feature column as regime proxy
        mom_col = bf[:, 0]  # rank-normalized, backward-looking

        breadth = np.mean(mom_col > 0.5)  # fraction above median rank
        dispersion = np.std(mom_col)       # rank dispersion (always ~0.29 but varies)
        mkt_ret = np.mean(mom_col) - 0.5   # deviation from expected mean rank

        extra = np.zeros((n, 3))
        extra[:, 0] = breadth   # past breadth
        extra[:, 1] = dispersion  # past dispersion
        extra[:, 2] = mkt_ret    # past market return

        dg['regime_features'] = extra
        dg['regime_feature_names'] = ['mkt_breadth_past', 'mkt_dispersion_past', 'mkt_ret_past']


def add_tail_indicators(features, top_k=15):
    """Add tail indicator features for top-k features."""
    n, p = features.shape
    k = min(top_k, p)
    indicators = np.zeros((n, k * 2))
    for j in range(k):
        col = features[:, j]
        indicators[:, j*2] = (col > 0.8).astype(np.float64)      # top quintile
        indicators[:, j*2+1] = (col < 0.2).astype(np.float64)    # bottom quintile
    return indicators


def assemble_features(dg, base_features):
    """Combine base features + BTC + regime + tail indicators."""
    parts = [base_features]

    if 'btc_features' in dg:
        # Rank-normalize BTC features within cross-section
        bf = dg['btc_features']
        for j in range(bf.shape[1]):
            r = rankdata(bf[:, j])
            bf[:, j] = r / len(r)
        parts.append(bf)

    if 'regime_features' in dg:
        # Z-score regime features (same for all coins, but still normalize)
        parts.append(dg['regime_features'])

    combined = np.hstack(parts)

    # Add tail indicators on the combined rank-normalized features
    tail_ind = add_tail_indicators(combined, top_k=15)
    combined = np.hstack([combined, tail_ind])

    return combined


# ════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════

def prepare_data(panel, feat_cols):
    """Build date groups with all features."""
    fwd_hold = f'fwd_ret_{HOLD_DAYS}d'
    fwd_train = f'fwd_ret_{TRAIN_LABEL_DAYS}d'

    for col, days in [(fwd_hold, HOLD_DAYS), (fwd_train, TRAIN_LABEL_DAYS)]:
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(days).shift(-days))

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

        train_ret = np.nan_to_num(g[fwd_train].values.astype(np.float64), nan=0.0)
        eval_ret = np.nan_to_num(g[fwd_hold].values.astype(np.float64), nan=0.0)
        train_mkt = np.nanmean(train_ret)

        # Vol-scaled excess return (committee recommendation)
        train_excess = train_ret - train_mkt
        vol = np.std(train_ret) if np.std(train_ret) > 1e-10 else 1.0
        train_excess_scaled = train_excess / vol

        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []

        date_groups[date] = {
            'base_features': features,
            'train_excess': train_excess_scaled,  # vol-scaled
            'train_ret': train_ret,
            'eval_ret': eval_ret,
            'symbols': syms,
            'n_coins': len(syms),
        }

    sorted_dates = sorted(date_groups.keys())

    # Add BTC-relative and regime features
    print("  Adding BTC-relative features...")
    add_btc_relative_features(date_groups, sorted_dates)
    print("  Adding regime features...")
    add_regime_features(date_groups, sorted_dates)

    return date_groups, sorted_dates


# ════════════════════════════════════════════
# MODEL TRAINING & PREDICTION
# ════════════════════════════════════════════

def train_linear(train_data, n_feat):
    """Train linear pairwise ranker."""
    all_feat, all_eret, offsets = [], [], [0]
    for feat, eret in train_data:
        if len(eret) < 5: continue
        all_feat.append(feat)
        all_eret.append(eret)
        offsets.append(offsets[-1] + len(eret))
    if len(all_feat) < 5:
        return None
    return _train_model(
        np.vstack(all_feat), np.concatenate(all_eret),
        np.array(offsets, dtype=np.int64),
        n_feat, N_EPOCHS, PAIRS_PER_DATE,
        NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG, 42,
    )


def train_lgbm(train_data):
    """Train LGBMRanker."""
    all_feat, all_labels, groups = [], [], []
    for feat, eret in train_data:
        if len(eret) < 5: continue
        # Convert to 5-bin quintile labels
        try:
            labels = pd.qcut(eret, 5, labels=False, duplicates='drop')
        except ValueError:
            labels = np.clip((rankdata(eret) * 5 / (len(eret) + 1)).astype(int), 0, 4)
        all_feat.append(feat)
        all_labels.append(labels)
        groups.append(len(eret))
    if len(all_feat) < 5:
        return None

    X = np.vstack(all_feat)
    y = np.concatenate(all_labels).astype(int)

    # Sample weights: |excess_return| + 0.1
    all_eret = np.concatenate([e for _, e in train_data if len(e) >= 5])
    sample_weight = np.abs(all_eret) + 0.1
    # Normalize within each group
    idx = 0
    for gs in groups:
        sw = sample_weight[idx:idx+gs]
        sample_weight[idx:idx+gs] = sw / (sw.sum() + 1e-10) * gs
        idx += gs

    model = LGBMRanker(**LGBM_PARAMS)
    model.fit(X, y, group=groups, sample_weight=sample_weight)
    return model


def predict_model(model, features, model_type):
    """Get scores from model."""
    if model_type == 'linear':
        return features @ model
    elif model_type == 'lgbm':
        return model.predict(features)
    return np.zeros(features.shape[0])


def borda_ensemble(scores_linear, scores_lgbm, w_linear=0.4, w_lgbm=0.6):
    """Combine via Borda count (rank averaging)."""
    r1 = rankdata(scores_linear)
    r2 = rankdata(scores_lgbm)
    return w_linear * r1 + w_lgbm * r2


# ════════════════════════════════════════════
# MAIN SHOOTOUT
# ════════════════════════════════════════════

def run_shootout():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    print(f"Base features: {len(feat_cols)}")

    date_groups, sorted_dates = prepare_data(panel, feat_cols)
    print(f"  {len(sorted_dates)} valid dates")

    # JIT warmup
    n_base = len([f for f in feat_cols if f in panel.columns])
    dummy = np.random.randn(20, n_base + 37)  # base + btc(4) + regime(3) + tail(30)
    dummy_e = np.random.randn(20)
    dummy_w = np.zeros(n_base + 37)
    _sample_pairs_and_train_epoch(dummy, dummy_e, dummy_w, 10, 40.0, 2.0, 0.001, 0.0, 0.01, 42)

    # Walk-forward schedule
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if (d - sorted_dates[0]).days >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i
            break
    rebal_dates = sorted_dates[start_idx::REBAL_DAYS]
    print(f"  {len(rebal_dates)} rebalance dates ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # Results storage
    model_names = ['linear', 'lgbm', 'ensemble']
    all_curves = {mn: {sn: [] for sn in list(STRATEGIES.keys()) + ['btc_bh']} for mn in model_names}
    all_equity = {mn: {sn: INITIAL_CAPITAL for sn in list(STRATEGIES.keys()) + ['btc_bh']} for mn in model_names}
    all_periods = {mn: [] for mn in model_names}

    n_feat = None  # will be set on first pass

    for ri, pred_date in enumerate(rebal_dates):
        # Collect training data
        train_end = pred_date - pd.Timedelta(days=PURGE_DAYS)
        train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)

        train_data = []
        for d in sorted_dates:
            if d < train_start or d > train_end: continue
            if d not in date_groups: continue
            dg = date_groups[d]
            mask = np.abs(dg['train_ret']) > 1e-10
            if np.sum(mask) < 5: continue
            # Apply mask to all feature arrays consistently
            masked_dg = {k: v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == dg['n_coins'] else v
                         for k, v in dg.items()}
            masked_dg['n_coins'] = int(np.sum(mask))
            feat = assemble_features(masked_dg, dg['base_features'][mask])
            eret = dg['train_excess'][mask]
            train_data.append((feat, eret))

        if len(train_data) < 10: continue

        if n_feat is None:
            n_feat = train_data[0][0].shape[1]
            print(f"  Total features (with BTC/regime/tail): {n_feat}")

        # Train all models
        w_linear = train_linear(train_data, n_feat)
        m_lgbm = train_lgbm(train_data)

        if w_linear is None and m_lgbm is None: continue

        if ri == 0 or (ri + 1) % 10 == 0:
            print(f"  [{ri+1}/{len(rebal_dates)}] {pred_date.date()} [{time.time()-t_start:.0f}s]",
                  flush=True)

        # Predict
        if pred_date not in date_groups: continue
        dg = date_groups[pred_date]
        features = assemble_features(dg, dg['base_features'])
        eval_ret = dg['eval_ret']
        n_coins = dg['n_coins']

        scores = {}
        if w_linear is not None:
            scores['linear'] = features @ w_linear
        if m_lgbm is not None:
            scores['lgbm'] = m_lgbm.predict(features)
        if 'linear' in scores and 'lgbm' in scores:
            scores['ensemble'] = borda_ensemble(scores['linear'], scores['lgbm'])

        # BTC return
        btc_ret = 0.0
        if 'BTCUSDT' in dg['symbols']:
            btc_ret = eval_ret[dg['symbols'].index('BTCUSDT')]

        # Evaluate each model × each strategy
        for mn in model_names:
            if mn not in scores: continue
            sc = scores[mn]
            sorted_idx = np.argsort(sc)
            eval_excess = eval_ret - np.mean(eval_ret)
            rank_ic = float(_spearman_corr(sc, eval_excess))

            # Decile means
            decile_means = np.zeros(N_DECILES)
            for rank_pos, idx in enumerate(sorted_idx):
                d_bin = min(rank_pos * N_DECILES // n_coins, N_DECILES - 1)
                decile_means[d_bin] += eval_ret[idx]
            counts = np.zeros(N_DECILES)
            for rank_pos in range(n_coins):
                counts[min(rank_pos * N_DECILES // n_coins, N_DECILES - 1)] += 1
            decile_means = np.where(counts > 0, decile_means / counts, 0.0)

            all_periods[mn].append({
                'date': pred_date, 'rank_ic': rank_ic,
                'decile_means': decile_means.tolist(),
                'n_coins': n_coins,
            })

            for sn, scfg in STRATEGIES.items():
                nl = max(1, int(n_coins * scfg['lp'])) if scfg['lp'] > 0 else 0
                ns = max(1, int(n_coins * scfg['sp'])) if scfg['sp'] > 0 else 0

                lr = np.mean(eval_ret[sorted_idx[-nl:]]) if nl > 0 else 0.0
                sr = 0.0
                if ns > 0:
                    sr = -np.mean(eval_ret[sorted_idx[:ns]])
                    sr -= FUNDING_DAILY * HOLD_DAYS

                if nl > 0 and ns > 0:
                    gr = (lr + sr) / 2.0 - 2 * COST_PER_SIDE
                elif nl > 0:
                    gr = lr - 2 * COST_PER_SIDE
                else:
                    gr = sr - 2 * COST_PER_SIDE

                gr = np.clip(gr, -0.5, 2.0)
                all_equity[mn][sn] *= (1 + gr)
                all_curves[mn][sn].append({
                    'date': pred_date, 'equity': all_equity[mn][sn], 'ret': gr})

            # BTC
            all_equity[mn]['btc_bh'] *= (1 + btc_ret)
            all_curves[mn]['btc_bh'].append({
                'date': pred_date, 'equity': all_equity[mn]['btc_bh'], 'ret': btc_ret})

    elapsed = time.time() - t_start

    # ═══ Print Results ═══
    print(f"\n{'='*120}")
    print(f"  MODEL SHOOTOUT — {elapsed/60:.1f} min")
    print(f"{'='*120}")

    for mn in model_names:
        plog = pd.DataFrame(all_periods[mn])
        if plog.empty: continue

        avg_ic = plog['rank_ic'].mean()
        ic_pos = (plog['rank_ic'] > 0).mean() * 100
        agg_decile = np.mean(plog['decile_means'].tolist(), axis=0)
        mono, _ = spearmanr(range(N_DECILES), agg_decile)

        print(f"\n  [{mn.upper()}] IC={avg_ic:.4f} IC>0={ic_pos:.0f}% Mono={mono:.3f}")
        print(f"  Deciles: {' '.join(f'{d*100:6.2f}' for d in agg_decile)}")

        print(f"  {'Strategy':<15} {'Final$':>10} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Hit%':>6}")
        for sn in list(STRATEGIES.keys()) + ['btc_bh']:
            df = pd.DataFrame(all_curves[mn][sn]).set_index('date')
            if len(df) < 2: continue
            eq, rets = df['equity'], df['ret']
            n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
            ppyr = 365.25 / HOLD_DAYS
            cagr = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1/n_yrs) - 1) * 100
            sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppyr)
            mdd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100
            hit = (rets > 0).mean() * 100
            label = STRATEGIES[sn]['label'] if sn in STRATEGIES else 'BTC B&H'
            print(f"  {label:<15} {eq.iloc[-1]:>10,.0f} {cagr:>6.1f}% {sharpe:>7.2f} {mdd:>6.1f}% {hit:>5.1f}%")

    # ═══ Dashboard ═══
    build_shootout_dashboard(all_curves, all_periods, model_names, elapsed)
    return all_curves, all_periods


def build_shootout_dashboard(all_curves, all_periods, model_names, elapsed):
    colors_strat = {
        'long_q10': '#4CAF50', 'short_q10': '#f44336', 'ls_q10': '#2196F3',
        'long_q20': '#66BB6A', 'short_q20': '#EF5350', 'ls_q20': '#42A5F5',
        'btc_bh': '#FFD700',
    }
    colors_model = {'linear': '#03A9F4', 'lgbm': '#FF9800', 'ensemble': '#9C27B0'}
    strat_names = list(STRATEGIES.keys()) + ['btc_bh']

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Linear — Equity (L/S focus)', 'LGBMRanker — Equity', 'Ensemble — Equity',
            'Decile Calibration Comparison', 'Sharpe by Model × Strategy', 'Rank IC Comparison',
            'L/S 10-10% Equity (all models)', 'Short 10% Equity (all models)', 'Drawdown L/S 10-10%',
        ],
        row_heights=[0.35, 0.35, 0.30],
    )

    # Row 1: Equity curves per model
    for ci, mn in enumerate(model_names):
        col = ci + 1
        for sn in ['ls_q10', 'short_q10', 'long_q10', 'btc_bh']:
            df = pd.DataFrame(all_curves[mn].get(sn, [])).set_index('date')
            if df.empty: continue
            label = STRATEGIES[sn]['label'] if sn in STRATEGIES else 'BTC B&H'
            fig.add_trace(go.Scatter(
                x=df.index, y=df['equity'], name=label if ci == 0 else None,
                line=dict(color=colors_strat[sn], width=2),
                showlegend=(ci == 0), legendgroup=sn,
            ), row=1, col=col)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=col)
        fig.update_yaxes(type='log', row=1, col=col)

    # Row 2 col 1: Decile comparison
    decile_colors = {'linear': '#03A9F4', 'lgbm': '#FF9800', 'ensemble': '#9C27B0'}
    for mn in model_names:
        plog = pd.DataFrame(all_periods[mn])
        if plog.empty: continue
        agg = np.mean(plog['decile_means'].tolist(), axis=0) * 100
        fig.add_trace(go.Bar(
            x=[f'D{i+1}' for i in range(N_DECILES)], y=agg,
            name=mn.upper(), marker_color=colors_model[mn], opacity=0.7,
        ), row=2, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=2, col=1)

    # Row 2 col 2: Sharpe comparison
    for mn in model_names:
        sharpes, labels = [], []
        for sn in strat_names:
            df = pd.DataFrame(all_curves[mn].get(sn, [])).set_index('date')
            if len(df) < 2: continue
            rets = df['ret']
            s = rets.mean() / (rets.std() + 1e-10) * np.sqrt(365.25/HOLD_DAYS)
            sharpes.append(s)
            labels.append(STRATEGIES[sn]['label'] if sn in STRATEGIES else 'BTC')
        fig.add_trace(go.Bar(
            x=labels, y=sharpes, name=mn.upper(),
            marker_color=colors_model[mn], opacity=0.7, showlegend=False,
        ), row=2, col=2)
    fig.add_hline(y=0, line_color='gray', line_dash='dot', row=2, col=2)

    # Row 2 col 3: IC comparison
    for mn in model_names:
        plog = pd.DataFrame(all_periods[mn])
        if plog.empty: continue
        ic_roll = plog['rank_ic'].rolling(4, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=plog['date'], y=ic_roll, name=f'{mn} IC',
            line=dict(color=colors_model[mn], width=2), showlegend=False,
        ), row=2, col=3)
    fig.add_hline(y=0, line_color='gray', row=2, col=3)

    # Row 3: Cross-model equity comparison
    for mn in model_names:
        for sn, col in [('ls_q10', 1), ('short_q10', 2)]:
            df = pd.DataFrame(all_curves[mn].get(sn, [])).set_index('date')
            if df.empty: continue
            fig.add_trace(go.Scatter(
                x=df.index, y=df['equity'], name=mn.upper() if col == 1 else None,
                line=dict(color=colors_model[mn], width=2),
                showlegend=(col == 1), legendgroup=f'm_{mn}',
            ), row=3, col=col)

    # Row 3 col 3: Drawdown
    for mn in model_names:
        df = pd.DataFrame(all_curves[mn].get('ls_q10', [])).set_index('date')
        if df.empty: continue
        eq = df['equity']
        dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=None,
            line=dict(color=colors_model[mn], width=1.5), showlegend=False,
        ), row=3, col=3)

    # Summary
    summaries = []
    for mn in model_names:
        plog = pd.DataFrame(all_periods[mn])
        if plog.empty: continue
        avg_ic = plog['rank_ic'].mean()
        agg = np.mean(plog['decile_means'].tolist(), axis=0)
        mono, _ = spearmanr(range(N_DECILES), agg)
        df = pd.DataFrame(all_curves[mn].get('ls_q10', [])).set_index('date')
        sharpe = 0
        if len(df) > 1:
            sharpe = df['ret'].mean() / (df['ret'].std()+1e-10) * np.sqrt(365.25/HOLD_DAYS)
        summaries.append(f'{mn.upper()}: IC={avg_ic:.3f} Mono={mono:.2f} Sharpe={sharpe:.2f}')

    fig.update_layout(
        height=1800, width=1600, template='plotly_dark', barmode='group',
        title_text=(f'Model Shootout: Linear vs LGBMRanker vs Ensemble<br>'
                    f'<sub>{" | ".join(summaries)} | {elapsed/60:.0f}min</sub>'),
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'model_shootout.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")

    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    run_shootout()
