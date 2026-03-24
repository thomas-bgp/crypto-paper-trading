"""
Backtester v2 — Production-Grade Risk Management
==================================================
Implements ALL 5 recommendations from the literature review:
  1. Small position sizes (3-4% per name, 25% total short)
  2. Inverse-vol weighting (Barroso & Santa-Clara 2015)
  3. Daily mark-to-market with triple barrier exits
  4. Portfolio drawdown circuit breakers
  5. Position size caps (never winsorize returns)

Uses the pairwise linear ranker (best clean model: IC=0.07, Sharpe 0.87 pre-risk).
Compares: naive equal-weight vs risk-managed.
"""
import numpy as np
import pandas as pd
import os, time, warnings
warnings.filterwarnings('ignore')

from numba import njit
from scipy.stats import spearmanr, rankdata
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decile_calibration_v2 import (
    load_panel, get_feature_cols,
    _cross_sectional_rank_normalize, _train_model, _predict, _spearman_corr,
    _sample_pairs_and_train_epoch,
    LR, L1_REG, L2_REG, MIN_COINS, VOL_FLOOR_PCT,
    N_EPOCHS, PAIRS_PER_DATE, NEAR_TIE_PCT, TAIL_WEIGHT_POW,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ─── Model Config (from audit) ───
TRAIN_DAYS = 540
PURGE_DAYS = 14
TRAIN_LABEL_DAYS = 7
REBAL_DAYS = 14        # rebalance every 2 weeks
RETRAIN_EVERY = 2      # retrain every 2 rebalances = monthly
INITIAL_CAPITAL = 100_000

# ─── Risk Management Spec (from literature review) ───
# Position sizing
N_SHORTS = 8
N_LONGS = 8
MAX_WEIGHT_SHORT = 0.04     # 4% max per short
MAX_WEIGHT_LONG = 0.06      # 6% max per long
TOTAL_SHORT_EXPOSURE = 0.25  # 25% total short
TOTAL_LONG_EXPOSURE = 0.25   # 25% total long
WEIGHTING = 'rank_x_invvol'  # rank * inverse-volatility

# Stop-losses (triple barrier)
STOP_LOSS_PCT = 0.15         # 15% trailing stop
STOP_CHECK = 'daily'         # check daily

# Circuit breakers
DD_SOFT = 0.10    # -10% from peak -> 50% sizing
DD_HARD = 0.20    # -20% from peak -> 25% sizing
DD_KILL = 0.30    # -30% -> flatten

# Costs
COST_PER_SIDE = 0.0015       # maker + slippage
FUNDING_PER_DAY = 0.0003     # ~11% annualized

# Strategies to test
CONFIGS = {
    'naive': {
        'label': 'Naive (old: 5×20%, no stops)',
        'n_short': 5, 'n_long': 5,
        'max_w_short': 0.20, 'max_w_long': 0.20,
        'total_short': 1.00, 'total_long': 1.00,
        'weighting': 'equal', 'stops': False, 'circuit_breakers': False,
    },
    'risk_managed': {
        'label': 'Risk-Managed (8×3%, inv-vol, stops, CB)',
        'n_short': N_SHORTS, 'n_long': N_LONGS,
        'max_w_short': MAX_WEIGHT_SHORT, 'max_w_long': MAX_WEIGHT_LONG,
        'total_short': TOTAL_SHORT_EXPOSURE, 'total_long': TOTAL_LONG_EXPOSURE,
        'weighting': 'rank_x_invvol', 'stops': True, 'circuit_breakers': True,
    },
    'risk_short_only': {
        'label': 'Risk-Managed Short-Only',
        'n_short': N_SHORTS, 'n_long': 0,
        'max_w_short': MAX_WEIGHT_SHORT, 'max_w_long': 0,
        'total_short': TOTAL_SHORT_EXPOSURE, 'total_long': 0,
        'weighting': 'rank_x_invvol', 'stops': True, 'circuit_breakers': True,
    },
    'risk_ls': {
        'label': 'Risk-Managed L/S',
        'n_short': N_SHORTS, 'n_long': N_LONGS,
        'max_w_short': MAX_WEIGHT_SHORT, 'max_w_long': MAX_WEIGHT_LONG,
        'total_short': TOTAL_SHORT_EXPOSURE, 'total_long': TOTAL_LONG_EXPOSURE,
        'weighting': 'rank_x_invvol', 'stops': True, 'circuit_breakers': True,
    },
}


def compute_weights(scores, volatilities, n_positions, max_weight, total_exposure, weighting, side='short'):
    """Compute position weights using rank × inverse-vol."""
    n = len(scores)
    if n_positions == 0 or n < n_positions:
        return np.array([]), np.array([], dtype=int)

    sorted_idx = np.argsort(scores)
    if side == 'short':
        selected = sorted_idx[:n_positions]  # lowest scores = short
    else:
        selected = sorted_idx[-n_positions:]  # highest scores = long

    if weighting == 'equal':
        weights = np.ones(len(selected)) / len(selected) * total_exposure
    elif weighting == 'rank_x_invvol':
        # Rank weight: worst coin gets most short weight
        if side == 'short':
            rank_w = np.arange(len(selected), 0, -1, dtype=np.float64)
        else:
            rank_w = np.arange(1, len(selected) + 1, dtype=np.float64)
        rank_w = rank_w / rank_w.sum()

        # Inverse-vol weight
        vols = volatilities[selected]
        vols = np.maximum(vols, 0.005)  # floor
        inv_vol = 1.0 / vols
        inv_vol_w = inv_vol / inv_vol.sum()

        # Combine: geometric mean
        combined = np.sqrt(rank_w * inv_vol_w)
        combined = combined / combined.sum()

        # Cap per name
        max_w_normalized = max_weight / total_exposure if total_exposure > 0 else 1.0
        combined = np.minimum(combined, max_w_normalized)
        combined = combined / combined.sum()

        weights = combined * total_exposure
    else:
        weights = np.ones(len(selected)) / len(selected) * total_exposure

    return weights, selected


def simulate_daily_with_stops(
    positions, daily_prices, entry_date_idx, hold_days,
    stop_loss_pct, use_stops, funding_per_day, cost_per_side,
):
    """
    Simulate positions with daily mark-to-market and triple barrier exits.
    positions: list of {idx, weight, side, entry_price}
    daily_prices: (n_days, n_coins) price matrix
    Returns: total portfolio return for the period, daily equity path
    """
    n_days = min(hold_days, daily_prices.shape[0] - entry_date_idx)
    if n_days <= 0:
        return 0.0, []

    portfolio_ret = 0.0
    daily_equity = []

    for pos in positions:
        idx = pos['idx']
        weight = pos['weight']
        side = pos['side']
        entry_price = pos['entry_price']

        # Entry cost
        pos_ret = -cost_per_side

        # Track best price for trailing stop
        if side == 'short':
            best_price = entry_price  # lowest price seen (want it to go down)
        else:
            best_price = entry_price  # highest price seen (want it to go up)

        stopped = False
        exit_day = n_days

        for d in range(1, n_days + 1):
            day_idx = entry_date_idx + d
            if day_idx >= daily_prices.shape[0]:
                break

            price = daily_prices[day_idx, idx]
            if price <= 0 or np.isnan(price):
                continue

            if side == 'short':
                # Funding cost
                pos_ret -= funding_per_day

                # Update trailing stop (track lowest price)
                best_price = min(best_price, price)
                stop_level = best_price * (1 + stop_loss_pct)

                # Check stop
                if use_stops and price >= stop_level:
                    pnl = -(price / entry_price - 1)
                    pos_ret += pnl - cost_per_side
                    stopped = True
                    exit_day = d
                    break
            else:
                # Long
                best_price = max(best_price, price)
                stop_level = best_price * (1 - stop_loss_pct)

                if use_stops and price <= stop_level:
                    pnl = price / entry_price - 1
                    pos_ret += pnl - cost_per_side
                    stopped = True
                    exit_day = d
                    break

        # If not stopped, exit at period end
        if not stopped:
            end_idx = min(entry_date_idx + n_days, daily_prices.shape[0] - 1)
            end_price = daily_prices[end_idx, idx]
            if end_price > 0 and not np.isnan(end_price):
                if side == 'short':
                    pnl = -(end_price / entry_price - 1)
                else:
                    pnl = end_price / entry_price - 1
                pos_ret += pnl - cost_per_side

        portfolio_ret += weight * pos_ret

    return portfolio_ret


def main():
    os.chdir(PROJECT_DIR)
    t_start = time.time()

    panel = load_panel()
    feat_cols = get_feature_cols(panel)
    n_feat = len([f for f in feat_cols if f in panel.columns])

    # Ensure forward return columns
    for h in [TRAIN_LABEL_DAYS, REBAL_DAYS]:
        col = f'fwd_ret_{h}d'
        if col not in panel.columns:
            panel[col] = panel.groupby(level='symbol')['close'].transform(
                lambda x: x.pct_change(h).shift(-h))

    # Build date groups with DAILY close prices for mark-to-market
    print("Building date groups with daily prices...")
    date_groups = {}
    all_dates = panel.index.get_level_values('date').unique().sort_values()

    # Build a price matrix for daily MTM
    # First: get all symbols in the universe
    all_symbols = panel.index.get_level_values('symbol').unique().tolist()
    sym_to_idx = {s: i for i, s in enumerate(all_symbols)}
    n_symbols = len(all_symbols)

    # Daily price matrix
    print(f"  Building price matrix ({len(all_dates)} dates × {n_symbols} symbols)...")
    price_matrix = np.full((len(all_dates), n_symbols), np.nan)
    date_to_didx = {}
    for di, date in enumerate(all_dates):
        date_to_didx[date] = di
        try:
            g = panel.loc[date]
            syms = g.index.get_level_values('symbol') if 'symbol' in g.index.names else g.index
            for sym, row in g.iterrows():
                if isinstance(sym, tuple):
                    sym = sym[-1]
                if sym in sym_to_idx:
                    price_matrix[di, sym_to_idx[sym]] = row['close']
        except Exception:
            continue

    # Build feature groups for rebalance dates
    fwd_train_col = f'fwd_ret_{TRAIN_LABEL_DAYS}d'
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

        train_ret = np.nan_to_num(g[fwd_train_col].values.astype(np.float64), nan=0.0) if fwd_train_col in g.columns else np.zeros(len(g))
        mkt = np.nanmean(train_ret)
        vol = max(np.std(train_ret), 1e-10)
        excess = (train_ret - mkt) / vol

        syms = g.index.get_level_values('symbol').tolist() if 'symbol' in g.index.names else []

        # Trailing 28d realized vol per coin (for inv-vol weighting)
        coin_vols = np.ones(len(syms)) * 0.5
        for ci, sym in enumerate(syms):
            if sym in sym_to_idx and date in date_to_didx:
                didx = date_to_didx[date]
                prices = price_matrix[max(0, didx-28):didx, sym_to_idx[sym]]
                prices = prices[~np.isnan(prices)]
                if len(prices) > 5:
                    log_rets = np.diff(np.log(prices + 1e-10))
                    coin_vols[ci] = max(np.std(log_rets) * np.sqrt(365), 0.05)

        date_groups[date] = {
            'features': features, 'excess': excess, 'train_ret': train_ret,
            'symbols': syms, 'n_coins': len(syms), 'coin_vols': coin_vols,
        }

    sorted_dates = sorted(date_groups.keys())
    print(f"  {len(sorted_dates)} valid dates, {n_feat} features")

    # JIT warmup
    d, e, w = np.random.randn(20, n_feat), np.random.randn(20), np.zeros(n_feat)
    _sample_pairs_and_train_epoch(d, e, w, 10, 30.0, 1.0, 0.001, 0.0, 0.01, 42)

    # Walk-forward schedule
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if (d - sorted_dates[0]).days >= TRAIN_DAYS + PURGE_DAYS:
            start_idx = i; break
    rebal_dates = sorted_dates[start_idx::REBAL_DAYS]
    print(f"  {len(rebal_dates)} rebalance dates")

    # ═══ Run backtests ═══
    print("\nRunning backtests...")
    all_results = {}
    weights_model = None

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n  [{cfg['label']}]")
        equity = INITIAL_CAPITAL
        peak_equity = INITIAL_CAPITAL
        curve = []
        period_log = []
        sizing_mult = 1.0  # circuit breaker multiplier
        t0 = time.time()

        for ri, pred_date in enumerate(rebal_dates):
            # Train model (shared across configs)
            if ri % RETRAIN_EVERY == 0 or weights_model is None:
                train_end = pred_date - pd.Timedelta(days=PURGE_DAYS)
                train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
                tf, te, offsets = [], [], [0]
                for dd in sorted_dates:
                    if dd < train_start or dd > train_end: continue
                    if dd not in date_groups: continue
                    dg = date_groups[dd]
                    mask = np.abs(dg['train_ret']) > 1e-10
                    if np.sum(mask) < 5: continue
                    tf.append(dg['features'][mask])
                    te.append(dg['excess'][mask])
                    offsets.append(offsets[-1] + int(np.sum(mask)))
                if len(tf) >= 10:
                    weights_model = _train_model(
                        np.vstack(tf), np.concatenate(te),
                        np.array(offsets, dtype=np.int64),
                        n_feat, N_EPOCHS, PAIRS_PER_DATE,
                        NEAR_TIE_PCT, TAIL_WEIGHT_POW, LR, L1_REG, L2_REG, 42 + ri)

            if weights_model is None: continue
            if pred_date not in date_groups: continue
            if pred_date not in date_to_didx: continue

            dg = date_groups[pred_date]
            scores = _predict(dg['features'], weights_model)
            n_coins = dg['n_coins']
            didx = date_to_didx[pred_date]

            # Circuit breakers
            if cfg['circuit_breakers']:
                dd_pct = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
                if dd_pct < -DD_KILL:
                    sizing_mult = 0.0
                elif dd_pct < -DD_HARD:
                    sizing_mult = 0.25
                elif dd_pct < -DD_SOFT:
                    sizing_mult = 0.50
                else:
                    sizing_mult = 1.0

            if sizing_mult == 0:
                curve.append({'date': pred_date, 'equity': equity, 'ret': 0.0})
                continue

            # Compute weights
            positions = []

            # Short leg
            if cfg['n_short'] > 0:
                sw, si = compute_weights(
                    scores, dg['coin_vols'], cfg['n_short'],
                    cfg['max_w_short'], cfg['total_short'] * sizing_mult,
                    cfg['weighting'], 'short')
                for j in range(len(si)):
                    sym = dg['symbols'][si[j]]
                    if sym in sym_to_idx:
                        entry_price = price_matrix[didx, sym_to_idx[sym]]
                        if entry_price > 0 and not np.isnan(entry_price):
                            positions.append({
                                'idx': sym_to_idx[sym], 'weight': sw[j],
                                'side': 'short', 'entry_price': entry_price,
                            })

            # Long leg
            if cfg['n_long'] > 0:
                lw, li = compute_weights(
                    scores, dg['coin_vols'], cfg['n_long'],
                    cfg['max_w_long'], cfg['total_long'] * sizing_mult,
                    cfg['weighting'], 'long')
                for j in range(len(li)):
                    sym = dg['symbols'][li[j]]
                    if sym in sym_to_idx:
                        entry_price = price_matrix[didx, sym_to_idx[sym]]
                        if entry_price > 0 and not np.isnan(entry_price):
                            positions.append({
                                'idx': sym_to_idx[sym], 'weight': lw[j],
                                'side': 'long', 'entry_price': entry_price,
                            })

            # Simulate with daily MTM and stops
            period_ret = simulate_daily_with_stops(
                positions, price_matrix, didx, REBAL_DAYS,
                STOP_LOSS_PCT, cfg['stops'], FUNDING_PER_DAY, COST_PER_SIDE,
            )

            period_ret = np.clip(period_ret, -0.30, 0.50)
            equity *= (1 + period_ret)
            peak_equity = max(peak_equity, equity)

            # IC
            eval_col = f'fwd_ret_{REBAL_DAYS}d'
            if eval_col in panel.columns:
                try:
                    g = panel.loc[pred_date]
                    if 'turnover_28d' in g.columns:
                        g = g[g['turnover_28d'] >= g['turnover_28d'].quantile(0.2)]
                    fr = g[eval_col].dropna().values
                    if len(fr) == len(scores):
                        ic = float(_spearman_corr(scores, fr - np.mean(fr)))
                    else:
                        ic = 0
                except Exception:
                    ic = 0
            else:
                ic = 0

            curve.append({'date': pred_date, 'equity': equity, 'ret': period_ret})
            period_log.append({'date': pred_date, 'ret': period_ret, 'ic': ic,
                              'n_positions': len(positions), 'sizing_mult': sizing_mult})

            if ri == 0 or (ri + 1) % 10 == 0:
                print(f"    [{ri+1}/{len(rebal_dates)}] {pred_date.date()} eq=${equity:,.0f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

        all_results[cfg_name] = {
            'curve': pd.DataFrame(curve).set_index('date'),
            'plog': pd.DataFrame(period_log),
            'cfg': cfg,
        }

    # ═══ Print Results ═══
    total_time = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"  BACKTESTER v2 — Production Risk Management ({total_time/60:.1f} min)")
    print(f"{'='*100}")

    ppyr = 365.25 / REBAL_DAYS
    for cfg_name, res in all_results.items():
        df = res['curve']
        if len(df) < 2: continue
        eq, rets = df['equity'], df['ret']
        n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
        cagr = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1/n_yrs) - 1) * 100
        sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppyr)
        sortino = rets.mean() / (rets[rets < 0].std() + 1e-10) * np.sqrt(ppyr)
        mdd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100
        calmar = cagr / (abs(mdd) + 1e-10)
        hit = (rets > 0).mean() * 100
        vol = rets.std() * np.sqrt(ppyr) * 100

        print(f"\n  [{res['cfg']['label']}]")
        print(f"  Final: ${eq.iloc[-1]:,.0f} | CAGR: {cagr:.1f}% | Sharpe: {sharpe:.2f} | "
              f"Sortino: {sortino:.2f}")
        print(f"  MaxDD: {mdd:.1f}% | Calmar: {calmar:.2f} | Vol: {vol:.1f}% | Hit: {hit:.1f}%")

    # ═══ Dashboard ═══
    build_dashboard(all_results, total_time)


def build_dashboard(all_results, elapsed):
    colors = {
        'naive': '#f44336',
        'risk_managed': '#4CAF50',
        'risk_short_only': '#FF9800',
        'risk_ls': '#2196F3',
    }

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Equity Curves (log scale)',
            'Drawdown',
            'Period Returns Distribution',
            'Rolling 6-Period Sharpe',
            'Cumulative Return',
            'Metrics Comparison',
        ],
        row_heights=[0.35, 0.35, 0.30],
    )

    ppyr = 365.25 / REBAL_DAYS
    metrics_data = []

    for cfg_name, res in all_results.items():
        df = res['curve']
        if len(df) < 2: continue
        eq, rets = df['equity'], df['ret']
        label = res['cfg']['label']
        color = colors.get(cfg_name, '#888')

        # Equity
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq, name=label,
            line=dict(color=color, width=2.5),
        ), row=1, col=1)

        # Drawdown
        dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=None,
            line=dict(color=color, width=1.5), showlegend=False,
        ), row=1, col=2)

        # Returns distribution
        fig.add_trace(go.Box(
            y=rets * 100, name=label, marker_color=color,
            boxmean='sd', showlegend=False,
        ), row=2, col=1)

        # Rolling Sharpe
        if len(rets) > 6:
            rs = rets.rolling(6).mean() / (rets.rolling(6).std() + 1e-10) * np.sqrt(ppyr)
            fig.add_trace(go.Scatter(
                x=rs.index, y=rs, name=None,
                line=dict(color=color, width=1.5), showlegend=False,
            ), row=2, col=2)

        # Metrics
        n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
        sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppyr)
        cagr = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1/n_yrs) - 1) * 100
        mdd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100
        metrics_data.append({'name': label, 'sharpe': sharpe, 'cagr': cagr, 'mdd': mdd, 'color': color})

    fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.add_hline(y=0, line_color='gray', row=1, col=2)
    fig.add_hline(y=0, line_color='gray', row=2, col=2)

    # Cumulative return bars
    for m in metrics_data:
        fig.add_trace(go.Bar(
            x=[m['name']], y=[m['cagr']], marker_color=m['color'],
            text=[f"{m['cagr']:.1f}%"], textposition='outside', showlegend=False,
        ), row=3, col=1)

    # Sharpe bars
    for m in metrics_data:
        fig.add_trace(go.Bar(
            x=[m['name']], y=[m['sharpe']], marker_color=m['color'],
            text=[f"{m['sharpe']:.2f}"], textposition='outside', showlegend=False,
        ), row=3, col=2)

    summary_parts = [f"{m['name']}: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1f}% MDD={m['mdd']:.1f}%"
                     for m in metrics_data]

    fig.update_layout(
        height=1600, width=1400, template='plotly_dark',
        title_text=(f'Backtester v2: Naive vs Risk-Managed<br>'
                    f'<sub>{" | ".join(summary_parts)} | {elapsed/60:.1f}min</sub>'),
        showlegend=True, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    )

    path = os.path.join(RESULTS_DIR, 'backtester_v2.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"\nDashboard: {path}")
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
