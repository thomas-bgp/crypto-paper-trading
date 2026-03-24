"""
Backtester v3 — CatBoost + Cross-Asset Features + Risk Management
===================================================================
Fixes ALL identified problems:
  1. CatBoost YetiRank (tree interactions)
  2. RAW excess return labels (no vol-scaling)
  3. ALL 216 features + cross-asset features (no pre-filtering, model selects)
  4. Train window = 180 days
  5. Universe = top 100 by volume
  6. Cross-asset features: correlation with BTC, with market, pairwise dispersion
  7. Full risk management (inv-vol, stops, circuit breakers)
  8. PCA fallback if too many features
"""
import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRanker
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, rankdata
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _spearman_corr,
    _sample_pairs_and_train_epoch, _train_model, _predict,
    LR, L1_REG, L2_REG, MIN_COINS, N_EPOCHS, PAIRS_PER_DATE,
    NEAR_TIE_PCT, TAIL_WEIGHT_POW,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ─── Config ───
TRAIN_DAYS = 180
PURGE_DAYS = 14
TRAIN_LABEL_DAYS = 7
REBAL_DAYS = 14
RETRAIN_EVERY = 2
INITIAL_CAPITAL = 100_000
TOP_N_UNIVERSE = 100       # filter to top 100 by volume
USE_PCA = True
PCA_COMPONENTS = 50

# Risk management
N_SHORTS = 8
N_LONGS = 8
MAX_W_SHORT = 0.04
MAX_W_LONG = 0.06
TOTAL_SHORT = 0.25
TOTAL_LONG = 0.25
STOP_LOSS = 0.15
COST_PER_SIDE = 0.0015
FUNDING_PER_DAY = 0.0003
DD_SOFT = 0.10
DD_HARD = 0.20

# CatBoost params
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

CONFIGS = {
    'catboost_risk': {
        'label': 'CatBoost + Risk Mgmt',
        'model': 'catboost', 'risk': True,
    },
    'linear_risk': {
        'label': 'Linear Pairwise + Risk Mgmt',
        'model': 'linear', 'risk': True,
    },
    'catboost_naive': {
        'label': 'CatBoost Naive (5x20%)',
        'model': 'catboost', 'risk': False,
    },
}

STRATEGIES = ['short_only', 'ls', 'long_only']
STRAT_LABELS = {'short_only': 'Short-Only', 'ls': 'L/S', 'long_only': 'Long-Only'}


# ════════════════════════════════════════════
# CROSS-ASSET FEATURES
# ════════════════════════════════════════════

def compute_cross_asset_features(panel, feat_cols):
    """Compute features that capture cross-asset relationships."""
    print("  Computing cross-asset features...")

    # Get daily returns per coin
    ret_col = 'fwd_ret_7d'  # backward-looking 7d return already in features
    # Use ret_7d from features (backward-looking)
    if 'ret_7d' not in panel.columns:
        panel['ret_7d'] = panel.groupby(level='symbol')['close'].pct_change(7)

    # 1. Rolling correlation with BTC (28d)
    print("    BTC correlation...")
    btc_ret = None
    if 'BTCUSDT' in panel.index.get_level_values('symbol').unique():
        btc_data = panel.xs('BTCUSDT', level='symbol')['ret_7d'] if 'ret_7d' in panel.columns else None
        if btc_data is not None:
            btc_ret = btc_data.reindex(panel.index.get_level_values('date').unique())

    cross_features = {}

    for date, group in panel.groupby(level='date'):
        n = len(group)
        syms = group.index.get_level_values('symbol').tolist() if 'symbol' in group.index.names else []

        # Get recent returns for each coin (from features already computed)
        feats = {}

        # a) Coin return vs market mean (already have as excess, but add explicitly)
        if 'ret_7d' in group.columns:
            r7 = group['ret_7d'].values
            mkt_mean = np.nanmean(r7)
            mkt_std = np.nanstd(r7) if np.nanstd(r7) > 1e-10 else 1.0
            feats['ret_vs_market'] = (r7 - mkt_mean)
            feats['ret_zscore_xs'] = (r7 - mkt_mean) / mkt_std

        # b) Market breadth (same for all coins on this date)
        if 'ret_7d' in group.columns:
            breadth = np.nanmean(r7 > 0)
            feats['mkt_breadth'] = np.full(n, breadth)

        # c) Cross-sectional dispersion
        if 'ret_7d' in group.columns:
            dispersion = np.nanstd(r7)
            feats['xs_dispersion'] = np.full(n, dispersion)

        # d) Coin's rank in the cross-section (momentum rank)
        if 'ret_7d' in group.columns:
            valid = ~np.isnan(r7)
            rank = np.full(n, 0.5)
            if valid.sum() > 3:
                rank[valid] = rankdata(r7[valid]) / valid.sum()
            feats['momentum_rank'] = rank

        # e) Correlation with BTC (use rolling 28d if available)
        if btc_ret is not None and 'ret_7d' in group.columns:
            feats['btc_corr_proxy'] = np.full(n, 0.5)
            # Simple proxy: how much does this coin's recent return track BTC?
            btc_r = btc_ret.get(date, 0)
            if abs(btc_r) > 1e-6:
                feats['btc_corr_proxy'] = np.where(
                    np.abs(r7) > 1e-6,
                    np.clip(r7 / btc_r, -3, 3) / 3 * 0.5 + 0.5,
                    0.5)

        # f) Relative strength vs BTC
        if btc_ret is not None and 'ret_7d' in group.columns:
            btc_r = btc_ret.get(date, 0)
            feats['rs_vs_btc'] = r7 - btc_r

        # g) Volatility rank in cross-section
        if 'rvol_28d' in group.columns:
            v28 = group['rvol_28d'].values
            valid = ~np.isnan(v28)
            vrank = np.full(n, 0.5)
            if valid.sum() > 3:
                vrank[valid] = rankdata(v28[valid]) / valid.sum()
            feats['vol_rank_xs'] = vrank

        # h) Volume rank
        if 'vol_avg_28d' in group.columns:
            va = group['vol_avg_28d'].values
            valid = ~np.isnan(va)
            varank = np.full(n, 0.5)
            if valid.sum() > 3:
                varank[valid] = rankdata(va[valid]) / valid.sum()
            feats['volume_rank_xs'] = varank

        cross_features[date] = feats

    return cross_features


# ════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════

def prepare_data(panel, feat_cols):
    """Build date groups with cross-asset features, top-N universe filter."""
    fwd_train = f'fwd_ret_{TRAIN_LABEL_DAYS}d'
    fwd_eval = f'fwd_ret_{REBAL_DAYS}d'

    for col, h in [(fwd_train, TRAIN_LABEL_DAYS), (fwd_eval, REBAL_DAYS)]:
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(h).shift(-h))

    # Cross-asset features
    cross_feats = compute_cross_asset_features(panel, feat_cols)

    # Build price matrix for daily MTM
    all_dates = panel.index.get_level_values('date').unique().sort_values()
    all_symbols = panel.index.get_level_values('symbol').unique().tolist()
    sym_to_idx = {s: i for i, s in enumerate(all_symbols)}
    n_symbols = len(all_symbols)

    print(f"  Building price matrix ({len(all_dates)}×{n_symbols})...")
    price_matrix = np.full((len(all_dates), n_symbols), np.nan)
    date_to_didx = {}
    for di, date in enumerate(all_dates):
        date_to_didx[date] = di
        try:
            g = panel.loc[date]
            for sym, row in g.iterrows():
                s = sym[-1] if isinstance(sym, tuple) else sym
                if s in sym_to_idx:
                    price_matrix[di, sym_to_idx[s]] = row['close']
        except Exception:
            continue

    # Build date groups
    print("  Building date groups (top 100 universe)...")
    date_groups = {}
    for date, group in panel.groupby(level='date'):
        g = group.dropna(subset=['close'])
        if len(g) < MIN_COINS: continue

        # TOP N UNIVERSE FILTER: keep only top 100 by 28d avg volume
        if 'vol_avg_28d' in g.columns:
            g = g.nlargest(TOP_N_UNIVERSE, 'vol_avg_28d')
        if len(g) < MIN_COINS: continue

        available = [f for f in feat_cols if f in g.columns]
        features = g[available].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Add cross-asset features
        if date in cross_feats:
            cf = cross_feats[date]
            # Need to align indices
            syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []
            all_syms_date = panel.loc[date].index.get_level_values('symbol').tolist() if 'symbol' in panel.loc[date].index.names else []

            cross_arr = []
            for fname in sorted(cf.keys()):
                full_arr = cf[fname]
                # Map from full cross-section to filtered universe
                col_vals = np.zeros(len(g))
                for ci, sym in enumerate(syms):
                    if sym in all_syms_date:
                        orig_idx = all_syms_date.index(sym)
                        if orig_idx < len(full_arr):
                            col_vals[ci] = full_arr[orig_idx]
                cross_arr.append(col_vals)

            if cross_arr:
                cross_matrix = np.column_stack(cross_arr)
                cross_matrix = np.nan_to_num(cross_matrix, nan=0.0)
                features = np.hstack([features, cross_matrix])

        # Rank-normalize ALL features
        features = _cross_sectional_rank_normalize(features)

        # Labels: RAW excess return (NO vol-scaling)
        train_ret = np.nan_to_num(g[fwd_train].values.astype(np.float64), nan=0.0) if fwd_train in g.columns else np.zeros(len(g))
        eval_ret = np.nan_to_num(g[fwd_eval].values.astype(np.float64), nan=0.0) if fwd_eval in g.columns else np.zeros(len(g))
        train_excess = train_ret - np.nanmean(train_ret)  # RAW excess, no vol-scaling

        # Coin vols for inv-vol weighting
        coin_vols = np.ones(len(g)) * 0.5
        for ci, sym in enumerate(syms):
            if sym in sym_to_idx and date in date_to_didx:
                didx = date_to_didx[date]
                prices = price_matrix[max(0, didx-28):didx, sym_to_idx[sym]]
                prices = prices[~np.isnan(prices)]
                if len(prices) > 5:
                    lr = np.diff(np.log(prices + 1e-10))
                    coin_vols[ci] = max(np.std(lr) * np.sqrt(365), 0.05)

        btc_ret = eval_ret[syms.index('BTCUSDT')] if 'BTCUSDT' in syms else 0.0

        date_groups[date] = {
            'features': features, 'train_excess': train_excess,
            'train_ret': train_ret, 'eval_ret': eval_ret,
            'symbols': syms, 'n_coins': len(syms), 'coin_vols': coin_vols,
            'btc_ret': btc_ret,
        }

    sorted_dates = sorted(date_groups.keys())
    n_feat = date_groups[sorted_dates[0]]['features'].shape[1] if sorted_dates else 0
    print(f"  {len(sorted_dates)} valid dates, {n_feat} total features (base + cross-asset)")

    return date_groups, sorted_dates, price_matrix, date_to_didx, sym_to_idx, n_feat


# ════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════

def train_catboost(train_data, n_feat):
    """Train CatBoost YetiRank."""
    all_X, all_y, group_ids = [], [], []
    gid = 0
    for feat, excess in train_data:
        if len(excess) < 5: continue
        # Quintile labels for ranking
        try:
            labels = pd.qcut(excess, 5, labels=False, duplicates='drop')
        except ValueError:
            labels = np.clip((rankdata(excess) * 5 / (len(excess) + 1)).astype(int), 0, 4)
        all_X.append(feat)
        all_y.append(labels)
        group_ids.extend([gid] * len(labels))
        gid += 1

    if len(all_X) < 5:
        return None

    X = np.vstack(all_X)
    y = np.concatenate(all_y).astype(int)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    model = CatBoostRanker(**CATBOOST_PARAMS)
    model.fit(X, y, group_id=np.array(group_ids))
    return model


def train_linear(train_data, n_feat):
    """Train linear pairwise ranker."""
    tf, te, offsets = [], [], [0]
    for feat, excess in train_data:
        if len(excess) < 5: continue
        tf.append(feat)
        te.append(excess)
        offsets.append(offsets[-1] + len(excess))
    if len(tf) < 5: return None
    return _train_model(
        np.vstack(tf), np.concatenate(te),
        np.array(offsets, dtype=np.int64),
        n_feat, N_EPOCHS, PAIRS_PER_DATE,
        NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG, 42)


# ════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════

def compute_weights(scores, vols, n_pos, max_w, total_exp, side, use_risk):
    """Compute position weights."""
    n = len(scores)
    if n_pos == 0 or n < n_pos: return np.array([]), np.array([], dtype=int)

    si = np.argsort(scores)
    sel = si[:n_pos] if side == 'short' else si[-n_pos:]

    if not use_risk:
        w = np.ones(len(sel)) / len(sel)
        return w, sel

    # Rank × inverse-vol
    if side == 'short':
        rw = np.arange(len(sel), 0, -1, dtype=np.float64)
    else:
        rw = np.arange(1, len(sel) + 1, dtype=np.float64)
    rw /= rw.sum()

    v = np.maximum(vols[sel], 0.01)
    iv = (1.0 / v); iv /= iv.sum()

    combined = np.sqrt(rw * iv)
    combined /= combined.sum()
    combined = np.minimum(combined, max_w / total_exp)
    combined /= combined.sum()

    return combined * total_exp, sel


def simulate_period(positions, price_matrix, entry_didx, hold_days, use_stops):
    """Daily MTM with triple barrier stops."""
    n_days = min(hold_days, price_matrix.shape[0] - entry_didx - 1)
    if n_days <= 0: return 0.0

    total_ret = 0.0
    for pos in positions:
        idx, weight, side = pos['idx'], pos['weight'], pos['side']
        ep = pos['entry_price']
        if ep <= 0 or np.isnan(ep): continue

        ret = -COST_PER_SIDE
        best = ep
        stopped = False

        for d in range(1, n_days + 1):
            di = entry_didx + d
            if di >= price_matrix.shape[0]: break
            p = price_matrix[di, idx]
            if p <= 0 or np.isnan(p): continue

            if side == 'short':
                ret -= FUNDING_PER_DAY
                best = min(best, p)
                if use_stops and p >= best * (1 + STOP_LOSS):
                    ret += -(p / ep - 1) - COST_PER_SIDE
                    stopped = True; break
            else:
                best = max(best, p)
                if use_stops and p <= best * (1 - STOP_LOSS):
                    ret += (p / ep - 1) - COST_PER_SIDE
                    stopped = True; break

        if not stopped:
            end_di = min(entry_didx + n_days, price_matrix.shape[0] - 1)
            end_p = price_matrix[end_di, idx]
            if end_p > 0 and not np.isnan(end_p):
                pnl = -(end_p / ep - 1) if side == 'short' else (end_p / ep - 1)
                ret += pnl - COST_PER_SIDE

        total_ret += weight * ret

    return total_ret


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

def main():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)

    date_groups, sorted_dates, price_matrix, date_to_didx, sym_to_idx, n_feat = \
        prepare_data(panel, feat_cols)

    # PCA if requested
    pca_model = None
    if USE_PCA and n_feat > PCA_COMPONENTS:
        print(f"  Fitting PCA: {n_feat} -> {PCA_COMPONENTS} components...")
        # Fit on a sample of training data
        sample_feats = []
        for d in sorted_dates[:500]:
            if d in date_groups:
                sample_feats.append(date_groups[d]['features'])
        if sample_feats:
            X_sample = np.vstack(sample_feats)
            X_sample = np.nan_to_num(X_sample, nan=0.0)
            pca_model = PCA(n_components=PCA_COMPONENTS, random_state=42)
            pca_model.fit(X_sample)
            explained = pca_model.explained_variance_ratio_.sum()
            print(f"  PCA explains {explained*100:.1f}% of variance")

            # Transform all date groups
            for d in sorted_dates:
                if d in date_groups:
                    date_groups[d]['features'] = pca_model.transform(
                        np.nan_to_num(date_groups[d]['features'], nan=0.0))
            n_feat = PCA_COMPONENTS

    # JIT warmup
    d, e, w = np.random.randn(20, n_feat), np.random.randn(20), np.zeros(n_feat)
    _sample_pairs_and_train_epoch(d, e, w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # Walk-forward
    start_idx = 0
    for i, dd in enumerate(sorted_dates):
        if (dd - sorted_dates[0]).days >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i; break
    rebal_dates = sorted_dates[start_idx::REBAL_DAYS]
    print(f"  {len(rebal_dates)} rebalance dates")

    # Results
    all_results = {}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n  === {cfg['label']} ===")
        use_risk = cfg['risk']
        ns = N_SHORTS if use_risk else 5
        nl = N_LONGS if use_risk else 5
        mws = MAX_W_SHORT if use_risk else 0.20
        mwl = MAX_W_LONG if use_risk else 0.20
        ts = TOTAL_SHORT if use_risk else 1.0
        tl = TOTAL_LONG if use_risk else 1.0

        strat_equity = {s: INITIAL_CAPITAL for s in STRATEGIES}
        strat_curves = {s: [] for s in STRATEGIES}
        strat_peak = {s: INITIAL_CAPITAL for s in STRATEGIES}
        period_log = []
        model = None
        t0 = time.time()

        for ri, pred_date in enumerate(rebal_dates):
            if ri % RETRAIN_EVERY == 0 or model is None:
                train_end = pred_date - pd.Timedelta(days=PURGE_DAYS)
                train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
                td = [(date_groups[d]['features'][np.abs(date_groups[d]['train_ret']) > 1e-10],
                       date_groups[d]['train_excess'][np.abs(date_groups[d]['train_ret']) > 1e-10])
                      for d in sorted_dates
                      if train_start <= d <= train_end and d in date_groups
                      and np.sum(np.abs(date_groups[d]['train_ret']) > 1e-10) >= 5]

                if len(td) < 10: continue

                if cfg['model'] == 'catboost':
                    model = train_catboost(td, n_feat)
                else:
                    model = train_linear(td, n_feat)

                if ri == 0 or (ri+1) % 10 == 0:
                    print(f"    [{ri+1}/{len(rebal_dates)}] {pred_date.date()} [{time.time()-t0:.0f}s]",
                          flush=True)

            if model is None: continue
            if pred_date not in date_groups or pred_date not in date_to_didx: continue

            dg = date_groups[pred_date]
            didx = date_to_didx[pred_date]
            X = np.nan_to_num(dg['features'], nan=0.0)

            if cfg['model'] == 'catboost':
                scores = model.predict(X)
            else:
                scores = X @ model
            scores = (scores - scores.mean()) / (scores.std() + 1e-10)

            nc = dg['n_coins']
            vols = dg['coin_vols']
            er = dg['eval_ret']
            ic = float(_spearman_corr(scores, er - np.mean(er)))

            for strat in STRATEGIES:
                # Circuit breaker
                sizing = 1.0
                if use_risk:
                    dd = (strat_equity[strat] - strat_peak[strat]) / strat_peak[strat]
                    if dd < -DD_HARD: sizing = 0.25
                    elif dd < -DD_SOFT: sizing = 0.5

                positions = []
                if strat in ('short_only', 'ls'):
                    sw, si = compute_weights(scores, vols, ns, mws, ts * sizing, 'short', use_risk)
                    for j in range(len(si)):
                        sym = dg['symbols'][si[j]]
                        if sym in sym_to_idx:
                            ep = price_matrix[didx, sym_to_idx[sym]]
                            if ep > 0 and not np.isnan(ep):
                                positions.append({'idx': sym_to_idx[sym], 'weight': sw[j],
                                                'side': 'short', 'entry_price': ep})

                if strat in ('long_only', 'ls'):
                    lw, li = compute_weights(scores, vols, nl, mwl, tl * sizing, 'long', use_risk)
                    for j in range(len(li)):
                        sym = dg['symbols'][li[j]]
                        if sym in sym_to_idx:
                            ep = price_matrix[didx, sym_to_idx[sym]]
                            if ep > 0 and not np.isnan(ep):
                                positions.append({'idx': sym_to_idx[sym], 'weight': lw[j],
                                                'side': 'long', 'entry_price': ep})

                ret = simulate_period(positions, price_matrix, didx, REBAL_DAYS, use_risk)
                ret = np.clip(ret, -0.30, 0.50)
                strat_equity[strat] *= (1 + ret)
                strat_peak[strat] = max(strat_peak[strat], strat_equity[strat])
                strat_curves[strat].append({'date': pred_date, 'equity': strat_equity[strat], 'ret': ret})

            period_log.append({'date': pred_date, 'ic': ic, 'n': nc})

        all_results[cfg_name] = {
            'curves': {s: pd.DataFrame(strat_curves[s]).set_index('date') for s in STRATEGIES},
            'plog': pd.DataFrame(period_log),
            'cfg': cfg,
        }

        # Print
        plog = pd.DataFrame(period_log)
        avg_ic = plog['ic'].mean() if len(plog) > 0 else 0
        print(f"    IC={avg_ic:.4f} IC>0={(plog['ic']>0).mean()*100:.0f}%")
        ppyr = 365.25 / REBAL_DAYS
        for strat in STRATEGIES:
            df = pd.DataFrame(strat_curves[strat]).set_index('date')
            if len(df) < 2: continue
            eq, rets = df['equity'], df['ret']
            n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
            cagr = ((eq.iloc[-1]/INITIAL_CAPITAL)**(1/n_yrs)-1)*100
            sharpe = rets.mean()/(rets.std()+1e-10)*np.sqrt(ppyr)
            mdd = ((eq-eq.expanding().max())/eq.expanding().max()).min()*100
            hit = (rets>0).mean()*100
            print(f"    {STRAT_LABELS[strat]:<12} ${eq.iloc[-1]:>9,.0f} CAGR={cagr:>5.1f}% "
                  f"Sharpe={sharpe:>5.2f} MDD={mdd:>5.1f}% Hit={hit:.0f}%")

    elapsed = time.time() - t_start
    build_dashboard(all_results, elapsed)


def build_dashboard(all_results, elapsed):
    colors_cfg = {'catboost_risk': '#4CAF50', 'linear_risk': '#2196F3', 'catboost_naive': '#f44336'}
    colors_strat = {'short_only': '#FF9800', 'ls': '#9C27B0', 'long_only': '#03A9F4'}

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'CatBoost+Risk — Equity', 'Linear+Risk — Equity', 'CatBoost Naive — Equity',
            'Sharpe Comparison', 'Rank IC Over Time', 'CAGR Comparison',
            'Short-Only Across Models', 'L/S Across Models', 'Drawdown (Short-Only)',
        ],
        row_heights=[0.35, 0.35, 0.30],
    )

    # Row 1: Equity per config
    for ci, (cfg_name, res) in enumerate(all_results.items()):
        col = ci + 1
        for strat in STRATEGIES:
            df = res['curves'][strat]
            if df.empty: continue
            fig.add_trace(go.Scatter(
                x=df.index, y=df['equity'], name=STRAT_LABELS[strat] if ci == 0 else None,
                line=dict(color=colors_strat[strat], width=2),
                showlegend=(ci == 0), legendgroup=strat,
            ), row=1, col=col)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=col)
        fig.update_yaxes(type='log', row=1, col=col)

    # Row 2: Comparisons
    ppyr = 365.25 / REBAL_DAYS
    for cfg_name, res in all_results.items():
        sharpes, cagrs, labels = [], [], []
        for strat in STRATEGIES:
            df = res['curves'][strat]
            if len(df) < 2: continue
            rets = df['ret']
            sharpes.append(rets.mean()/(rets.std()+1e-10)*np.sqrt(ppyr))
            n_yrs = max((df.index[-1]-df.index[0]).days/365.25, 0.1)
            cagrs.append(((df['equity'].iloc[-1]/INITIAL_CAPITAL)**(1/n_yrs)-1)*100)
            labels.append(STRAT_LABELS[strat])
        fig.add_trace(go.Bar(x=labels, y=sharpes, name=res['cfg']['label'],
            marker_color=colors_cfg[cfg_name], opacity=0.8), row=2, col=1)
        fig.add_trace(go.Bar(x=labels, y=cagrs, name=None, showlegend=False,
            marker_color=colors_cfg[cfg_name], opacity=0.8), row=2, col=3)

    # IC
    for cfg_name, res in all_results.items():
        pl = res['plog']
        if pl.empty: continue
        ic_roll = pl['ic'].rolling(4, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=pl['date'], y=ic_roll, name=res['cfg']['label'],
            line=dict(color=colors_cfg[cfg_name], width=2), showlegend=False), row=2, col=2)
    fig.add_hline(y=0, line_color='gray', row=2, col=2)

    # Row 3: Cross-model
    for cfg_name, res in all_results.items():
        for strat, col in [('short_only', 1), ('ls', 2)]:
            df = res['curves'][strat]
            if df.empty: continue
            fig.add_trace(go.Scatter(x=df.index, y=df['equity'],
                name=res['cfg']['label'] if col == 1 else None,
                line=dict(color=colors_cfg[cfg_name], width=2),
                showlegend=(col == 1), legendgroup=f'm_{cfg_name}'), row=3, col=col)

        df = res['curves']['short_only']
        if not df.empty:
            eq = df['equity']
            dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
            fig.add_trace(go.Scatter(x=dd.index, y=dd, name=None,
                line=dict(color=colors_cfg[cfg_name], width=1.5),
                showlegend=False), row=3, col=3)

    fig.update_layout(
        height=1800, width=1600, template='plotly_dark', barmode='group',
        title_text=f'v3: CatBoost + Cross-Asset + Risk Management<br><sub>{elapsed/60:.1f}min</sub>',
        showlegend=True, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'backtester_v3.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
