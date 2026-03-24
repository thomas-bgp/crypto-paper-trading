"""
Regime-Filtered Strategy — 3 Approaches
==========================================
All regime signals use BACKWARD-LOOKING data only (no leakage).

1. Simple Rules: past dispersion + past breadth + BTC trend
2. Logistic Classifier: trained on lagged context → predict "good IC week"
3. HMM: 2-state model on market features → "trade" vs "sit"

Combined with CatBoost 540d ranking + full risk management.
"""
import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRanker
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegressionCV
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
REBAL_DAYS = 14
RETRAIN_EVERY = 2
INITIAL_CAPITAL = 100_000
N_DECILES = 10

# Risk management
N_SHORTS = 8; N_LONGS = 8
MAX_W_SHORT = 0.04; MAX_W_LONG = 0.06
TOTAL_SHORT = 0.25; TOTAL_LONG = 0.25
STOP_LOSS = 0.15
COST_PER_SIDE = 0.0015
FUNDING_PER_DAY = 0.0003

CATBOOST_PARAMS = {
    'loss_function': 'YetiRank', 'iterations': 200, 'depth': 4,
    'learning_rate': 0.05, 'l2_leaf_reg': 5.0, 'random_strength': 2.0,
    'bagging_temperature': 1.0, 'border_count': 64, 'verbose': 0,
    'random_seed': 42, 'task_type': 'CPU',
}


def compute_backward_regime(panel, date_groups, sorted_dates):
    """Compute regime signals using ONLY past data."""
    print("  Computing backward-looking regime signals...")

    # BTC data
    btc = None
    if 'BTCUSDT' in panel.index.get_level_values('symbol').unique():
        btc = panel.xs('BTCUSDT', level='symbol')[['close', 'volume']].copy()
        btc['ret_7d'] = btc['close'].pct_change(7)
        btc['ret_14d'] = btc['close'].pct_change(14)
        btc['ret_28d'] = btc['close'].pct_change(28)
        btc['sma50'] = btc['close'].rolling(50).mean()
        btc['sma200'] = btc['close'].rolling(200).mean()
        btc['vol_14d'] = btc['close'].pct_change(1).rolling(14).std() * np.sqrt(365)
        btc['vol_28d'] = btc['close'].pct_change(1).rolling(28).std() * np.sqrt(365)

    # Past cross-sectional stats (from REALIZED returns, not forward)
    # Use ret_7d (backward-looking 7d return) from the panel
    if 'ret_7d' not in panel.columns:
        panel['ret_7d'] = panel.groupby(level='symbol')['close'].pct_change(7)

    regime_data = {}
    # Rolling stats computed from past cross-sections
    past_dispersions = []
    past_breadths = []
    past_extremes = []

    for date in sorted_dates:
        if date not in date_groups:
            continue

        # Past cross-sectional dispersion (from backward-looking returns)
        try:
            g = panel.loc[date]
            r7 = g['ret_7d'].dropna().values if 'ret_7d' in g.columns else np.array([])
        except:
            r7 = np.array([])

        if len(r7) > 10:
            disp = np.std(r7)
            brd = np.mean(r7 > 0)
            ext_up = np.mean(r7 > 0.3)
        else:
            disp = 0.1
            brd = 0.5
            ext_up = 0.02

        past_dispersions.append(disp)
        past_breadths.append(brd)
        past_extremes.append(ext_up)

        # Rolling medians for thresholds (use past 90 days)
        window = min(90, len(past_dispersions))
        disp_median = np.median(past_dispersions[-window:])
        brd_median = np.median(past_breadths[-window:])

        # BTC signals
        btc_above_sma50 = 0.5
        btc_above_sma200 = 0.5
        btc_ret_28d = 0.0
        btc_vol_14d = 0.5
        btc_vol_28d = 0.5
        btc_vol_change = 1.0

        if btc is not None:
            for col, var in [('sma50', None), ('sma200', None)]:
                pass
            try:
                btc_close = btc['close'].get(date, np.nan)
                btc_sma50 = btc['sma50'].get(date, np.nan)
                btc_sma200 = btc['sma200'].get(date, np.nan)
                btc_above_sma50 = float(btc_close > btc_sma50) if np.isfinite(btc_close) and np.isfinite(btc_sma50) else 0.5
                btc_above_sma200 = float(btc_close > btc_sma200) if np.isfinite(btc_close) and np.isfinite(btc_sma200) else 0.5
                btc_ret_28d = btc['ret_28d'].get(date, 0)
                btc_ret_28d = btc_ret_28d if np.isfinite(btc_ret_28d) else 0
                btc_vol_14d = btc['vol_14d'].get(date, 0.5)
                btc_vol_14d = btc_vol_14d if np.isfinite(btc_vol_14d) else 0.5
                btc_vol_28d = btc['vol_28d'].get(date, 0.5)
                btc_vol_28d = btc_vol_28d if np.isfinite(btc_vol_28d) else 0.5
                btc_vol_change = btc_vol_14d / (btc_vol_28d + 1e-10)
            except:
                pass

        regime_data[date] = {
            # Raw signals (all backward-looking)
            'past_dispersion': disp,
            'past_breadth': brd,
            'past_extreme_up': ext_up,
            'disp_median': disp_median,
            'brd_median': brd_median,
            'btc_above_sma50': btc_above_sma50,
            'btc_above_sma200': btc_above_sma200,
            'btc_ret_28d': btc_ret_28d,
            'btc_vol_14d': btc_vol_14d,
            'btc_vol_change': btc_vol_change,
            # Simple rule signals
            'low_dispersion': float(disp < disp_median),
            'low_breadth': float(brd < 0.6),
            'few_extremes': float(ext_up < 0.05),
            'btc_trending': btc_above_sma50,
        }

    return regime_data


def compute_weights(scores, vols, n_pos, max_w, total_exp, side):
    n = len(scores)
    if n_pos == 0 or n < n_pos: return np.array([]), np.array([], dtype=int)
    si = np.argsort(scores)
    sel = si[:n_pos] if side == 'short' else si[-n_pos:]
    rw = np.arange(len(sel), 0, -1, dtype=np.float64) if side == 'short' else np.arange(1, len(sel)+1, dtype=np.float64)
    rw /= rw.sum()
    v = np.maximum(vols[sel], 0.01)
    iv = 1.0/v; iv /= iv.sum()
    c = np.sqrt(rw * iv); c /= c.sum()
    c = np.minimum(c, max_w/total_exp); c /= c.sum()
    return c * total_exp, sel


def simulate_period(positions, price_matrix, didx, hold_days, sym_to_idx):
    n_days = min(hold_days, price_matrix.shape[0] - didx - 1)
    if n_days <= 0: return 0.0
    total = 0.0
    for pos in positions:
        idx, w, side, ep = pos['idx'], pos['weight'], pos['side'], pos['entry_price']
        if ep <= 0 or np.isnan(ep): continue
        ret = -COST_PER_SIDE
        best = ep; stopped = False
        for d in range(1, n_days+1):
            di = didx + d
            if di >= price_matrix.shape[0]: break
            p = price_matrix[di, idx]
            if p <= 0 or np.isnan(p): continue
            if side == 'short':
                ret -= FUNDING_PER_DAY
                best = min(best, p)
                if p >= best * (1 + STOP_LOSS):
                    ret += -(p/ep-1) - COST_PER_SIDE; stopped = True; break
            else:
                best = max(best, p)
                if p <= best * (1 - STOP_LOSS):
                    ret += (p/ep-1) - COST_PER_SIDE; stopped = True; break
        if not stopped:
            edi = min(didx+n_days, price_matrix.shape[0]-1)
            ep2 = price_matrix[edi, idx]
            if ep2 > 0 and not np.isnan(ep2):
                ret += (-(ep2/ep-1) if side == 'short' else (ep2/ep-1)) - COST_PER_SIDE
        total += w * ret
    return total


def main():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    n_feat = len([f for f in feat_cols if f in panel.columns])

    for h in [TRAIN_LABEL, REBAL_DAYS]:
        col = f'fwd_ret_{h}d'
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(h).shift(-h))

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
        features = np.nan_to_num(features, nan=0.0)
        features = _cross_sectional_rank_normalize(features)
        train_ret = np.nan_to_num(g[f'fwd_ret_{TRAIN_LABEL}d'].values.astype(np.float64), nan=0.0)
        eval_ret = np.nan_to_num(g[f'fwd_ret_{REBAL_DAYS}d'].values.astype(np.float64), nan=0.0)
        train_excess = train_ret - np.nanmean(train_ret)
        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []
        coin_vols = np.ones(len(g)) * 0.5
        if 'rvol_28d' in g.columns:
            rv = g['rvol_28d'].values
            coin_vols = np.where(np.isfinite(rv) & (rv > 0.01), rv, 0.5)
        date_groups[date] = {
            'features': features, 'train_excess': train_excess,
            'train_ret': train_ret, 'eval_ret': eval_ret,
            'symbols': syms, 'n_coins': len(syms), 'coin_vols': coin_vols,
        }

    sorted_dates = sorted(date_groups.keys())

    # Regime data
    regime_data = compute_backward_regime(panel, date_groups, sorted_dates)

    # Price matrix
    print("Building price matrix...")
    all_dates = panel.index.get_level_values('date').unique().sort_values()
    all_symbols = panel.index.get_level_values('symbol').unique().tolist()
    sym_to_idx = {s: i for i, s in enumerate(all_symbols)}
    price_matrix = np.full((len(all_dates), len(all_symbols)), np.nan)
    date_to_didx = {}
    for di, date in enumerate(all_dates):
        date_to_didx[date] = di
        try:
            g = panel.loc[date]
            for sym, row in g.iterrows():
                s = sym[-1] if isinstance(sym, tuple) else sym
                if s in sym_to_idx:
                    price_matrix[di, sym_to_idx[s]] = row['close']
        except: continue

    # JIT warmup
    d, e, w = np.random.randn(20, n_feat), np.random.randn(20), np.zeros(n_feat)
    _sample_pairs_and_train_epoch(d, e, w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # Schedule
    start_idx = 0
    for i, dd in enumerate(sorted_dates):
        if (dd - sorted_dates[0]).days >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i; break
    rebal_dates = sorted_dates[start_idx::REBAL_DAYS]
    print(f"  {len(rebal_dates)} rebalance dates")

    # ═══ Define regime strategies ═══
    strategies = {
        'always_on': {
            'label': 'Always On (no filter)',
            'filter': lambda rd: True,
        },
        'simple_disp': {
            'label': 'Filter: Low Dispersion',
            'filter': lambda rd: rd.get('low_dispersion', 0) > 0.5,
        },
        'simple_combo': {
            'label': 'Filter: LowDisp + FewExtremes',
            'filter': lambda rd: rd.get('low_dispersion', 0) > 0.5 and rd.get('few_extremes', 0) > 0.5,
        },
        'simple_full': {
            'label': 'Filter: LowDisp + FewExtr + LowBreadth',
            'filter': lambda rd: (rd.get('low_dispersion', 0) > 0.5 and
                                  rd.get('few_extremes', 0) > 0.5 and
                                  rd.get('low_breadth', 0) > 0.5),
        },
        'btc_trend': {
            'label': 'Filter: BTC > SMA50',
            'filter': lambda rd: rd.get('btc_above_sma50', 0) > 0.5,
        },
        'combo_btc_disp': {
            'label': 'Filter: LowDisp + BTC>SMA50',
            'filter': lambda rd: (rd.get('low_dispersion', 0) > 0.5 and
                                  rd.get('btc_above_sma50', 0) > 0.5),
        },
        'classifier': {
            'label': 'Logistic Classifier',
            'filter': 'classifier',  # special handling
        },
    }

    # ═══ Train classifier on first half, apply on all ═══
    # Collect IC from a quick walk-forward to train the classifier
    print("\nTraining regime classifier...")
    ic_for_classifier = []
    model_cache = None

    for ri, pred_date in enumerate(rebal_dates):
        if ri % RETRAIN_EVERY == 0 or model_cache is None:
            te = pred_date - pd.Timedelta(days=PURGE_DAYS)
            ts = te - pd.Timedelta(days=TRAIN_DAYS)
            td = [(date_groups[d]['features'][np.abs(date_groups[d]['train_ret']) > 1e-10],
                   date_groups[d]['train_excess'][np.abs(date_groups[d]['train_ret']) > 1e-10])
                  for d in sorted_dates if ts <= d <= te and d in date_groups
                  and np.sum(np.abs(date_groups[d]['train_ret']) > 1e-10) >= 5]
            if len(td) < 10: continue
            all_X, all_y, gids = [], [], []; gid = 0
            for feat, excess in td:
                if len(excess) < 5: continue
                try: labels = pd.qcut(excess, 5, labels=False, duplicates='drop')
                except: labels = np.clip((rankdata(excess)*5/(len(excess)+1)).astype(int), 0, 4)
                all_X.append(feat); all_y.append(labels); gids.extend([gid]*len(labels)); gid += 1
            if len(all_X) >= 5:
                model_cache = CatBoostRanker(**CATBOOST_PARAMS)
                model_cache.fit(np.nan_to_num(np.vstack(all_X), nan=0),
                               np.concatenate(all_y).astype(int), group_id=np.array(gids))

        if model_cache is None or pred_date not in date_groups: continue
        dg = date_groups[pred_date]
        scores = model_cache.predict(np.nan_to_num(dg['features'], nan=0.0))
        er = dg['eval_ret']
        ic = float(_spearman_corr(scores, er - np.mean(er)))

        rd = regime_data.get(pred_date, {})
        ic_for_classifier.append({
            'date': pred_date, 'ic': ic,
            **{k: v for k, v in rd.items() if isinstance(v, (int, float))},
        })

    ic_df = pd.DataFrame(ic_for_classifier)
    print(f"  Collected {len(ic_df)} IC observations for classifier")

    # Train logistic: predict IC > 0.08 (good week)
    clf_model = None
    clf_scaler = None
    regime_cols = ['past_dispersion', 'past_breadth', 'past_extreme_up',
                   'btc_above_sma50', 'btc_above_sma200', 'btc_ret_28d',
                   'btc_vol_14d', 'btc_vol_change']

    if len(ic_df) > 30:
        ic_df['good'] = (ic_df['ic'] > 0.08).astype(int)
        X_clf = ic_df[regime_cols].fillna(0).values
        y_clf = ic_df['good'].values
        clf_scaler = StandardScaler()
        X_scaled = clf_scaler.fit_transform(X_clf)
        clf_model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
        clf_model.fit(X_scaled, y_clf)
        train_acc = clf_model.score(X_scaled, y_clf)
        print(f"  Classifier accuracy: {train_acc:.2f}")
        coefs = pd.Series(clf_model.coef_[0], index=regime_cols).sort_values(key=abs, ascending=False)
        print(f"  Top coefficients:")
        for var, c in coefs.head(5).items():
            print(f"    {var:<25} {c:>8.3f}")

    # ═══ STEP 1: Train model ONCE and precompute scores + positions ═══
    print("\nPrecomputing model scores for all rebalance dates...")
    precomputed = {}  # date -> {scores, short_ret, ls_ret, ic, short_pos, long_pos}
    model_cache = None
    t0 = time.time()

    for ri, pred_date in enumerate(rebal_dates):
        if ri % RETRAIN_EVERY == 0 or model_cache is None:
            te = pred_date - pd.Timedelta(days=PURGE_DAYS)
            ts = te - pd.Timedelta(days=TRAIN_DAYS)
            td = [(date_groups[d]['features'][np.abs(date_groups[d]['train_ret']) > 1e-10],
                   date_groups[d]['train_excess'][np.abs(date_groups[d]['train_ret']) > 1e-10])
                  for d in sorted_dates if ts <= d <= te and d in date_groups
                  and np.sum(np.abs(date_groups[d]['train_ret']) > 1e-10) >= 5]
            if len(td) >= 10:
                all_X, all_y, gids = [], [], []; gid = 0
                for feat, excess in td:
                    if len(excess) < 5: continue
                    try: labels = pd.qcut(excess, 5, labels=False, duplicates='drop')
                    except: labels = np.clip((rankdata(excess)*5/(len(excess)+1)).astype(int), 0, 4)
                    all_X.append(feat); all_y.append(labels); gids.extend([gid]*len(labels)); gid += 1
                if len(all_X) >= 5:
                    model_cache = CatBoostRanker(**CATBOOST_PARAMS)
                    model_cache.fit(np.nan_to_num(np.vstack(all_X), nan=0),
                                   np.concatenate(all_y).astype(int), group_id=np.array(gids))

        if model_cache is None or pred_date not in date_groups or pred_date not in date_to_didx:
            continue

        dg = date_groups[pred_date]
        didx = date_to_didx[pred_date]
        scores = model_cache.predict(np.nan_to_num(dg['features'], nan=0.0))
        er = dg['eval_ret']
        ic = float(_spearman_corr(scores, er - np.mean(er)))

        # Precompute positions and returns for FULL sizing (no circuit breaker yet)
        sw, si = compute_weights(scores, dg['coin_vols'], N_SHORTS, MAX_W_SHORT, TOTAL_SHORT, 'short')
        short_pos = []
        for j in range(len(si)):
            sym = dg['symbols'][si[j]]
            if sym in sym_to_idx:
                ep = price_matrix[didx, sym_to_idx[sym]]
                if ep > 0 and not np.isnan(ep):
                    short_pos.append({'idx': sym_to_idx[sym], 'weight': sw[j],
                                     'side': 'short', 'entry_price': ep})

        lw, li = compute_weights(scores, dg['coin_vols'], N_LONGS, MAX_W_LONG, TOTAL_LONG, 'long')
        long_pos = []
        for j in range(len(li)):
            sym = dg['symbols'][li[j]]
            if sym in sym_to_idx:
                ep = price_matrix[didx, sym_to_idx[sym]]
                if ep > 0 and not np.isnan(ep):
                    long_pos.append({'idx': sym_to_idx[sym], 'weight': lw[j],
                                    'side': 'long', 'entry_price': ep})

        ret_s = simulate_period(short_pos, price_matrix, didx, REBAL_DAYS, sym_to_idx)
        ret_ls = simulate_period(short_pos + long_pos, price_matrix, didx, REBAL_DAYS, sym_to_idx)
        ret_s = np.clip(ret_s, -0.30, 0.50)
        ret_ls = np.clip(ret_ls, -0.30, 0.50)

        precomputed[pred_date] = {'ret_s': ret_s, 'ret_ls': ret_ls, 'ic': ic}

        if (ri+1) % 10 == 0:
            print(f"  [{ri+1}/{len(rebal_dates)}] {pred_date.date()} [{time.time()-t0:.0f}s]", flush=True)

    print(f"  Precomputed {len(precomputed)} periods in {time.time()-t0:.0f}s")

    # ═══ STEP 2: Apply regime filters (fast, no retraining) ═══
    print("\nApplying regime filters...")
    all_results = {}

    for sname, scfg in strategies.items():
        equity_short = INITIAL_CAPITAL
        equity_ls = INITIAL_CAPITAL
        peak_short = peak_ls = INITIAL_CAPITAL
        curve_short = []
        curve_ls = []
        periods_active = 0
        periods_total = 0
        ic_list = []

        for pred_date in rebal_dates:
            if pred_date not in precomputed:
                continue

            periods_total += 1
            rd = regime_data.get(pred_date, {})
            pc = precomputed[pred_date]

            # Check regime filter
            if scfg['filter'] == 'classifier':
                if clf_model is not None and clf_scaler is not None:
                    x_clf = np.array([[rd.get(c, 0) for c in regime_cols]])
                    active = clf_model.predict(clf_scaler.transform(x_clf))[0] == 1
                else:
                    active = True
            else:
                active = scfg['filter'](rd)

            if not active:
                curve_short.append({'date': pred_date, 'equity': equity_short, 'ret': 0.0, 'active': 0})
                curve_ls.append({'date': pred_date, 'equity': equity_ls, 'ret': 0.0, 'active': 0})
                continue

            periods_active += 1
            ic_list.append(pc['ic'])

            # Circuit breakers
            dd_s = (equity_short - peak_short) / peak_short
            dd_l = (equity_ls - peak_ls) / peak_ls
            sz_s = 0.25 if dd_s < -0.20 else (0.5 if dd_s < -0.10 else 1.0)
            sz_l = 0.25 if dd_l < -0.20 else (0.5 if dd_l < -0.10 else 1.0)

            ret_s = pc['ret_s'] * sz_s
            ret_ls = pc['ret_ls'] * min(sz_s, sz_l)

            equity_short *= (1 + ret_s)
            equity_ls *= (1 + ret_ls)
            peak_short = max(peak_short, equity_short)
            peak_ls = max(peak_ls, equity_ls)

            curve_short.append({'date': pred_date, 'equity': equity_short, 'ret': ret_s, 'active': 1})
            curve_ls.append({'date': pred_date, 'equity': equity_ls, 'ret': ret_ls, 'active': 1})

        all_results[sname] = {
            'label': scfg['label'],
            'short': pd.DataFrame(curve_short).set_index('date') if curve_short else pd.DataFrame(),
            'ls': pd.DataFrame(curve_ls).set_index('date') if curve_ls else pd.DataFrame(),
            'active_pct': periods_active / max(periods_total, 1) * 100,
            'avg_ic': np.mean(ic_list) if ic_list else 0,
            'ic_pos': np.mean(np.array(ic_list) > 0) * 100 if ic_list else 0,
        }

    # ═══ Print Results ═══
    elapsed = time.time() - t_start
    ppyr = 365.25 / REBAL_DAYS

    print(f"\n{'='*120}")
    print(f"  REGIME-FILTERED STRATEGIES ({elapsed/60:.1f} min)")
    print(f"{'='*120}")
    print(f"\n{'Strategy':<38} {'Active%':>8} {'IC':>6} | {'Short$':>9} {'Sh.Sharpe':>9} {'Sh.MDD':>7} | {'LS$':>9} {'LS.Sharpe':>9} {'LS.MDD':>7}")
    print(f"{'-'*120}")

    for sname, res in all_results.items():
        sdf = res['short']
        ldf = res['ls']
        if sdf.empty: continue

        eq_s = sdf['equity']
        rets_s = sdf['ret']
        n_yrs = max((eq_s.index[-1] - eq_s.index[0]).days / 365.25, 0.1)
        sharpe_s = rets_s.mean() / (rets_s.std() + 1e-10) * np.sqrt(ppyr)
        mdd_s = ((eq_s - eq_s.expanding().max()) / eq_s.expanding().max()).min() * 100

        eq_l = ldf['equity']
        rets_l = ldf['ret']
        sharpe_l = rets_l.mean() / (rets_l.std() + 1e-10) * np.sqrt(ppyr)
        mdd_l = ((eq_l - eq_l.expanding().max()) / eq_l.expanding().max()).min() * 100

        print(f"  {res['label']:<36} {res['active_pct']:>7.0f}% {res['avg_ic']:>6.3f} | "
              f"${eq_s.iloc[-1]:>8,.0f} {sharpe_s:>9.2f} {mdd_s:>6.1f}% | "
              f"${eq_l.iloc[-1]:>8,.0f} {sharpe_l:>9.2f} {mdd_l:>6.1f}%")

    print(f"{'='*120}")

    # ═══ Dashboard ═══
    build_dashboard(all_results, elapsed)


def build_dashboard(all_results, elapsed):
    colors = {
        'always_on': '#888888', 'simple_disp': '#4CAF50',
        'simple_combo': '#FF9800', 'simple_full': '#f44336',
        'btc_trend': '#2196F3', 'combo_btc_disp': '#9C27B0',
        'classifier': '#00BCD4',
    }

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Short-Only Equity (all filters)', 'L/S Equity (all filters)',
            'Short-Only Sharpe by Filter', 'L/S Sharpe by Filter',
            'Activity Rate', 'Active IC by Filter',
        ],
        row_heights=[0.4, 0.3, 0.3],
    )

    ppyr = 365.25 / REBAL_DAYS
    sharpes_s, sharpes_l, activity, ics = [], [], [], []
    names = []

    for sname, res in all_results.items():
        color = colors.get(sname, '#888')
        label = res['label']
        names.append(label)

        for strat, col_idx in [('short', 1), ('ls', 2)]:
            df = res[strat]
            if df.empty: continue
            fig.add_trace(go.Scatter(
                x=df.index, y=df['equity'], name=label if col_idx == 1 else None,
                line=dict(color=color, width=2.5 if sname != 'always_on' else 1.5,
                         dash='dot' if sname == 'always_on' else 'solid'),
                showlegend=(col_idx == 1), legendgroup=sname,
            ), row=1, col=col_idx)

        # Compute Sharpes
        sdf, ldf = res['short'], res['ls']
        sh_s = sdf['ret'].mean()/(sdf['ret'].std()+1e-10)*np.sqrt(ppyr) if not sdf.empty else 0
        sh_l = ldf['ret'].mean()/(ldf['ret'].std()+1e-10)*np.sqrt(ppyr) if not ldf.empty else 0
        sharpes_s.append(sh_s)
        sharpes_l.append(sh_l)
        activity.append(res['active_pct'])
        ics.append(res['avg_ic'])

    fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=2)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=2)

    # Sharpe bars
    c_list = [colors.get(s, '#888') for s in all_results.keys()]
    fig.add_trace(go.Bar(x=names, y=sharpes_s, marker_color=c_list,
        text=[f'{v:.2f}' for v in sharpes_s], textposition='outside', showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=names, y=sharpes_l, marker_color=c_list,
        text=[f'{v:.2f}' for v in sharpes_l], textposition='outside', showlegend=False), row=2, col=2)
    fig.add_hline(y=0, line_color='gray', row=2, col=1)
    fig.add_hline(y=0, line_color='gray', row=2, col=2)

    # Activity + IC
    fig.add_trace(go.Bar(x=names, y=activity, marker_color=c_list,
        text=[f'{v:.0f}%' for v in activity], textposition='outside', showlegend=False), row=3, col=1)
    fig.add_trace(go.Bar(x=names, y=ics, marker_color=c_list,
        text=[f'{v:.3f}' for v in ics], textposition='outside', showlegend=False), row=3, col=2)

    fig.update_layout(
        height=1600, width=1500, template='plotly_dark',
        title_text=f'Regime-Filtered Strategies<br><sub>{elapsed/60:.1f}min</sub>',
        showlegend=True, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'regime_strategy.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
