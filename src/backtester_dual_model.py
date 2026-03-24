"""
Dual-Model L/S — Specialist Short + Specialist Long
=====================================================
Model A: CatBoost on FULL universe (535 coins) → SHORT leg (identifies losers)
Model B: Linear on TOP 100 by volume → LONG leg (identifies winners)

Combined L/S with full risk management.
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
TRAIN_DAYS_SHORT = 540     # short model: more data to learn structural losers
TRAIN_DAYS_LONG = 180      # long model: recent data for winners
PURGE_DAYS = 14
TRAIN_LABEL_DAYS = 7
REBAL_DAYS = 14
RETRAIN_EVERY = 2
INITIAL_CAPITAL = 100_000
TOP_N_LONG = 100           # long universe: top 100 by volume
VOL_FLOOR_SHORT = 0.20     # short universe: drop bottom 20% by volume

# Risk management
N_SHORTS = 10
N_LONGS = 10
MAX_W_SHORT = 0.035
MAX_W_LONG = 0.05
TOTAL_SHORT = 0.25
TOTAL_LONG = 0.25
STOP_LOSS = 0.15
COST_PER_SIDE = 0.0015
FUNDING_PER_DAY = 0.0003
DD_SOFT = 0.10
DD_HARD = 0.20

CATBOOST_PARAMS = {
    'loss_function': 'YetiRank', 'iterations': 200, 'depth': 4,
    'learning_rate': 0.05, 'l2_leaf_reg': 5.0, 'random_strength': 2.0,
    'bagging_temperature': 1.0, 'border_count': 64, 'verbose': 0,
    'random_seed': 42, 'task_type': 'CPU',
}


def build_universe(panel, feat_cols, mode):
    """Build date groups for a specific universe."""
    fwd_train = f'fwd_ret_{TRAIN_LABEL_DAYS}d'
    fwd_eval = f'fwd_ret_{REBAL_DAYS}d'

    for col, h in [(fwd_train, TRAIN_LABEL_DAYS), (fwd_eval, REBAL_DAYS)]:
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(h).shift(-h))

    date_groups = {}
    for date, group in panel.groupby(level='date'):
        g = group.dropna(subset=['close'])
        if len(g) < MIN_COINS: continue

        if mode == 'short_full':
            # Full universe, drop bottom 20% by volume
            if 'turnover_28d' in g.columns:
                thresh = g['turnover_28d'].quantile(VOL_FLOOR_SHORT)
                g = g[g['turnover_28d'] >= thresh]
        elif mode == 'long_top100':
            # Top 100 by volume
            if 'vol_avg_28d' in g.columns:
                g = g.nlargest(TOP_N_LONG, 'vol_avg_28d')

        if len(g) < MIN_COINS: continue

        available = [f for f in feat_cols if f in g.columns]
        features = g[available].values.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = _cross_sectional_rank_normalize(features)

        train_ret = np.nan_to_num(g[fwd_train].values.astype(np.float64), nan=0.0)
        eval_ret = np.nan_to_num(g[fwd_eval].values.astype(np.float64), nan=0.0)
        train_excess = train_ret - np.nanmean(train_ret)  # RAW excess

        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []

        # Coin vols
        coin_vols = np.ones(len(g)) * 0.5
        if 'rvol_28d' in g.columns:
            rv = g['rvol_28d'].values
            coin_vols = np.where(np.isfinite(rv) & (rv > 0.01), rv, 0.5)

        date_groups[date] = {
            'features': features, 'train_excess': train_excess,
            'train_ret': train_ret, 'eval_ret': eval_ret,
            'symbols': syms, 'n_coins': len(syms), 'coin_vols': coin_vols,
        }

    return date_groups


def train_catboost(train_data):
    """Train CatBoost YetiRank on list of (features, excess) tuples."""
    all_X, all_y, gids = [], [], []
    gid = 0
    for feat, excess in train_data:
        if len(excess) < 5: continue
        try:
            labels = pd.qcut(excess, 5, labels=False, duplicates='drop')
        except ValueError:
            labels = np.clip((rankdata(excess) * 5 / (len(excess) + 1)).astype(int), 0, 4)
        all_X.append(feat)
        all_y.append(labels)
        gids.extend([gid] * len(labels))
        gid += 1
    if len(all_X) < 5: return None
    X = np.nan_to_num(np.vstack(all_X), nan=0)
    y = np.concatenate(all_y).astype(int)
    model = CatBoostRanker(**CATBOOST_PARAMS)
    model.fit(X, y, group_id=np.array(gids))
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


def compute_weights(scores, vols, n_pos, max_w, total_exp, side):
    """Inv-vol × rank weighting."""
    n = len(scores)
    if n_pos == 0 or n < n_pos: return np.array([]), np.array([], dtype=int)
    si = np.argsort(scores)
    sel = si[:n_pos] if side == 'short' else si[-n_pos:]
    rw = np.arange(len(sel), 0, -1, dtype=np.float64) if side == 'short' else np.arange(1, len(sel)+1, dtype=np.float64)
    rw /= rw.sum()
    v = np.maximum(vols[sel], 0.01)
    iv = 1.0 / v; iv /= iv.sum()
    c = np.sqrt(rw * iv); c /= c.sum()
    c = np.minimum(c, max_w / total_exp); c /= c.sum()
    return c * total_exp, sel


def simulate_period(positions, price_matrix, entry_didx, hold_days):
    """Daily MTM with trailing stops."""
    n_days = min(hold_days, price_matrix.shape[0] - entry_didx - 1)
    if n_days <= 0: return 0.0
    total = 0.0
    for pos in positions:
        idx, w, side, ep = pos['idx'], pos['weight'], pos['side'], pos['entry_price']
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
                if p >= best * (1 + STOP_LOSS):
                    ret += -(p/ep - 1) - COST_PER_SIDE
                    stopped = True; break
            else:
                best = max(best, p)
                if p <= best * (1 - STOP_LOSS):
                    ret += (p/ep - 1) - COST_PER_SIDE
                    stopped = True; break
        if not stopped:
            end_di = min(entry_didx + n_days, price_matrix.shape[0] - 1)
            end_p = price_matrix[end_di, idx]
            if end_p > 0 and not np.isnan(end_p):
                pnl = -(end_p/ep-1) if side == 'short' else (end_p/ep-1)
                ret += pnl - COST_PER_SIDE
        total += w * ret
    return total


def main():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    n_feat = len([f for f in feat_cols if f in panel.columns])

    # Build both universes
    print("Building SHORT universe (full, 535 coins)...")
    dg_short = build_universe(panel, feat_cols, 'short_full')
    print(f"  {len(dg_short)} dates")

    print("Building LONG universe (top 100)...")
    dg_long = build_universe(panel, feat_cols, 'long_top100')
    print(f"  {len(dg_long)} dates")

    # Price matrix for daily MTM
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
        except Exception:
            continue

    # JIT warmup
    d, e, w = np.random.randn(20, n_feat), np.random.randn(20), np.zeros(n_feat)
    _sample_pairs_and_train_epoch(d, e, w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    sorted_dates = sorted(set(dg_short.keys()) & set(dg_long.keys()))
    start_idx = 0
    for i, dd in enumerate(sorted_dates):
        if (dd - sorted_dates[0]).days >= max(TRAIN_DAYS_SHORT, TRAIN_DAYS_LONG) + PURGE_DAYS:
            start_idx = i; break
    rebal_dates = sorted_dates[start_idx::REBAL_DAYS]
    print(f"  {len(rebal_dates)} rebalance dates")

    # ═══ Strategies ═══
    strats = {
        'dual_ls':       {'label': 'Dual L/S (specialist short + specialist long)', 'short': True, 'long': True},
        'specialist_short': {'label': 'Specialist Short-Only', 'short': True, 'long': False},
        'specialist_long':  {'label': 'Specialist Long-Only', 'short': False, 'long': True},
        'single_catboost_ls': {'label': 'Single CatBoost L/S (full universe)', 'short': True, 'long': True, 'single': True},
    }

    equity = {s: INITIAL_CAPITAL for s in strats}
    peak = {s: INITIAL_CAPITAL for s in strats}
    curves = {s: [] for s in strats}
    period_log = []

    model_short = None
    model_long = None
    model_single = None
    t0 = time.time()

    for ri, pred_date in enumerate(rebal_dates):
        if ri % RETRAIN_EVERY == 0 or model_short is None:
            # Train SHORT model (CatBoost on full universe, 540d)
            te_s = pred_date - pd.Timedelta(days=PURGE_DAYS)
            ts_s = te_s - pd.Timedelta(days=TRAIN_DAYS_SHORT)
            td_short = [(dg_short[d]['features'][np.abs(dg_short[d]['train_ret']) > 1e-10],
                         dg_short[d]['train_excess'][np.abs(dg_short[d]['train_ret']) > 1e-10])
                        for d in sorted_dates if ts_s <= d <= te_s and d in dg_short
                        and np.sum(np.abs(dg_short[d]['train_ret']) > 1e-10) >= 5]
            if len(td_short) >= 10:
                model_short = train_catboost(td_short)

            # Train LONG model (Linear on top 100, 180d)
            te_l = pred_date - pd.Timedelta(days=PURGE_DAYS)
            ts_l = te_l - pd.Timedelta(days=TRAIN_DAYS_LONG)
            td_long = [(dg_long[d]['features'][np.abs(dg_long[d]['train_ret']) > 1e-10],
                        dg_long[d]['train_excess'][np.abs(dg_long[d]['train_ret']) > 1e-10])
                       for d in sorted_dates if ts_l <= d <= te_l and d in dg_long
                       and np.sum(np.abs(dg_long[d]['train_ret']) > 1e-10) >= 5]
            if len(td_long) >= 10:
                model_long = train_linear(td_long, n_feat)

            # Single model for comparison (CatBoost on full)
            model_single = model_short

            if ri == 0 or (ri+1) % 10 == 0:
                print(f"  [{ri+1}/{len(rebal_dates)}] {pred_date.date()} "
                      f"short_train={len(td_short)} long_train={len(td_long)} [{time.time()-t0:.0f}s]",
                      flush=True)

        if pred_date not in date_to_didx: continue
        didx = date_to_didx[pred_date]

        # Get scores
        scores_short = None
        scores_long = None

        if model_short is not None and pred_date in dg_short:
            ds = dg_short[pred_date]
            X = np.nan_to_num(ds['features'], nan=0.0)
            scores_short = model_short.predict(X)
            scores_short = (scores_short - scores_short.mean()) / (scores_short.std() + 1e-10)

        if model_long is not None and pred_date in dg_long:
            dl = dg_long[pred_date]
            X = np.nan_to_num(dl['features'], nan=0.0)
            scores_long = X @ model_long
            scores_long = (scores_long - scores_long.mean()) / (scores_long.std() + 1e-10)

        # IC
        ic_short = 0
        ic_long = 0
        if scores_short is not None:
            ds = dg_short[pred_date]
            er = ds['eval_ret']
            ic_short = float(_spearman_corr(scores_short, er - np.mean(er)))
        if scores_long is not None:
            dl = dg_long[pred_date]
            er = dl['eval_ret']
            ic_long = float(_spearman_corr(scores_long, er - np.mean(er)))

        period_log.append({'date': pred_date, 'ic_short': ic_short, 'ic_long': ic_long})

        for sname, scfg in strats.items():
            # Circuit breaker
            dd = (equity[sname] - peak[sname]) / peak[sname]
            sizing = 1.0
            if dd < -DD_HARD: sizing = 0.25
            elif dd < -DD_SOFT: sizing = 0.5

            positions = []

            # SHORT leg
            if scfg.get('short') and scores_short is not None and pred_date in dg_short:
                ds = dg_short[pred_date]
                use_scores = scores_short
                if scfg.get('single'):
                    use_scores = scores_short

                sw, si = compute_weights(use_scores, ds['coin_vols'], N_SHORTS,
                                         MAX_W_SHORT, TOTAL_SHORT * sizing, 'short')
                for j in range(len(si)):
                    sym = ds['symbols'][si[j]]
                    if sym in sym_to_idx:
                        ep = price_matrix[didx, sym_to_idx[sym]]
                        if ep > 0 and not np.isnan(ep):
                            positions.append({'idx': sym_to_idx[sym], 'weight': sw[j],
                                            'side': 'short', 'entry_price': ep})

            # LONG leg
            if scfg.get('long'):
                if scfg.get('single') and scores_short is not None and pred_date in dg_short:
                    # Single model: use same model for long (from full universe)
                    ds = dg_short[pred_date]
                    lw, li = compute_weights(scores_short, ds['coin_vols'], N_LONGS,
                                             MAX_W_LONG, TOTAL_LONG * sizing, 'long')
                    for j in range(len(li)):
                        sym = ds['symbols'][li[j]]
                        if sym in sym_to_idx:
                            ep = price_matrix[didx, sym_to_idx[sym]]
                            if ep > 0 and not np.isnan(ep):
                                positions.append({'idx': sym_to_idx[sym], 'weight': lw[j],
                                                'side': 'long', 'entry_price': ep})
                elif scores_long is not None and pred_date in dg_long:
                    # Dual model: specialist long from top 100
                    dl = dg_long[pred_date]
                    lw, li = compute_weights(scores_long, dl['coin_vols'], N_LONGS,
                                             MAX_W_LONG, TOTAL_LONG * sizing, 'long')
                    for j in range(len(li)):
                        sym = dl['symbols'][li[j]]
                        if sym in sym_to_idx:
                            ep = price_matrix[didx, sym_to_idx[sym]]
                            if ep > 0 and not np.isnan(ep):
                                positions.append({'idx': sym_to_idx[sym], 'weight': lw[j],
                                                'side': 'long', 'entry_price': ep})

            ret = simulate_period(positions, price_matrix, didx, REBAL_DAYS)
            ret = np.clip(ret, -0.30, 0.50)
            equity[sname] *= (1 + ret)
            peak[sname] = max(peak[sname], equity[sname])
            curves[sname].append({'date': pred_date, 'equity': equity[sname], 'ret': ret})

    elapsed = time.time() - t_start

    # ═══ Print Results ═══
    print(f"\n{'='*100}")
    print(f"  DUAL-MODEL L/S ({elapsed/60:.1f} min)")
    print(f"{'='*100}")

    plog = pd.DataFrame(period_log)
    if len(plog) > 0:
        print(f"  Short Model IC: {plog['ic_short'].mean():.4f} (>0: {(plog['ic_short']>0).mean()*100:.0f}%)")
        print(f"  Long Model IC:  {plog['ic_long'].mean():.4f} (>0: {(plog['ic_long']>0).mean()*100:.0f}%)")

    ppyr = 365.25 / REBAL_DAYS
    metrics = {}
    print(f"\n  {'Strategy':<45} {'Final':>10} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MDD':>7} {'Hit':>5}")
    print(f"  {'-'*95}")
    for sname, scfg in strats.items():
        df = pd.DataFrame(curves[sname]).set_index('date')
        if len(df) < 2: continue
        eq, rets = df['equity'], df['ret']
        n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
        cagr = ((eq.iloc[-1]/INITIAL_CAPITAL)**(1/n_yrs)-1)*100
        sharpe = rets.mean()/(rets.std()+1e-10)*np.sqrt(ppyr)
        sortino = rets.mean()/(rets[rets<0].std()+1e-10)*np.sqrt(ppyr)
        mdd = ((eq-eq.expanding().max())/eq.expanding().max()).min()*100
        hit = (rets>0).mean()*100
        metrics[sname] = {'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd, 'sortino': sortino}
        print(f"  {scfg['label']:<45} ${eq.iloc[-1]:>9,.0f} {cagr:>6.1f}% {sharpe:>7.2f} "
              f"{sortino:>8.2f} {mdd:>6.1f}% {hit:>4.0f}%")
    print(f"{'='*100}")

    # ═══ Dashboard ═══
    colors = {
        'dual_ls': '#9C27B0',
        'specialist_short': '#f44336',
        'specialist_long': '#4CAF50',
        'single_catboost_ls': '#2196F3',
    }

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Equity Curves (log)', 'Drawdown',
            'Per-Period Returns', 'Rolling Sharpe (6-period)',
            'Sharpe Comparison', 'Rank IC: Short Model vs Long Model',
        ],
        row_heights=[0.35, 0.35, 0.30],
    )

    for sname, scfg in strats.items():
        df = pd.DataFrame(curves[sname]).set_index('date')
        if df.empty: continue
        eq, rets = df['equity'], df['ret']
        color = colors.get(sname, '#888')

        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=scfg['label'],
            line=dict(color=color, width=2.5)), row=1, col=1)

        dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=None,
            line=dict(color=color, width=1.5), showlegend=False), row=1, col=2)

        fig.add_trace(go.Box(y=rets*100, name=scfg['label'],
            marker_color=color, boxmean='sd', showlegend=False), row=2, col=1)

        if len(rets) > 6:
            rs = rets.rolling(6).mean()/(rets.rolling(6).std()+1e-10)*np.sqrt(ppyr)
            fig.add_trace(go.Scatter(x=rs.index, y=rs, name=None,
                line=dict(color=color, width=1.5), showlegend=False), row=2, col=2)

    fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.add_hline(y=0, line_color='gray', row=2, col=2)

    # Sharpe bars
    for sname in strats:
        if sname in metrics:
            fig.add_trace(go.Bar(x=[strats[sname]['label']], y=[metrics[sname]['sharpe']],
                marker_color=colors.get(sname, '#888'), showlegend=False,
                text=[f"{metrics[sname]['sharpe']:.2f}"], textposition='outside'), row=3, col=1)

    # IC comparison
    if len(plog) > 0:
        fig.add_trace(go.Scatter(x=plog['date'], y=plog['ic_short'].rolling(4,min_periods=1).mean(),
            name='Short Model IC', line=dict(color='#f44336', width=2)), row=3, col=2)
        fig.add_trace(go.Scatter(x=plog['date'], y=plog['ic_long'].rolling(4,min_periods=1).mean(),
            name='Long Model IC', line=dict(color='#4CAF50', width=2)), row=3, col=2)
        fig.add_hline(y=0, line_color='gray', row=3, col=2)

    best = max(metrics.items(), key=lambda x: x[1]['sharpe'])
    summary = f"BEST: {strats[best[0]]['label']} Sharpe={best[1]['sharpe']:.2f} CAGR={best[1]['cagr']:.1f}% MDD={best[1]['mdd']:.1f}%"

    fig.update_layout(
        height=1600, width=1400, template='plotly_dark',
        title_text=f'Dual-Model L/S: Specialist Short + Specialist Long<br><sub>{summary} | {elapsed/60:.1f}min</sub>',
        showlegend=True, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'dual_model_ls.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
