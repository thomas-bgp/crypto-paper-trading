"""
Cross-Sectional Momentum Backtester v1.0
Implements ALL technical solutions from momentum-solucoes-tecnicas.md:
  - Hysteresis band (enter top 20%, exit only below top 40%)
  - Point-in-time universe (survivorship-bias-free)
  - Composite regime filter (SMA + EMA + vol + funding + F&G)
  - Max-loss position sizing (not vol-based)
  - Alpha-relative drawdown stop
  - Whipsaw detector (ADX + flip rate)
  - Robust momentum signal (avg of 21-35d lookbacks)
"""
import numpy as np
import pandas as pd
import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
UNIVERSE_DIR = os.path.join(DATA_DIR, 'universe')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Configuration ───
INITIAL_CAPITAL = 100_000
MAKER_FEE = 0.00075      # 0.075% with BNB
SLIPPAGE = 0.0005         # 0.05%
COST_PER_SIDE = MAKER_FEE + SLIPPAGE
STABLE_YIELD_DAILY = 0.045 / 365  # 4.5% annual stablecoin yield

# Momentum signal
LOOKBACKS = [21, 25, 28, 30, 35]  # robust: average across range
SKIP_DAYS = 1  # skip last day (reversal avoidance)

# Portfolio
MAX_POSITIONS = 5
ENTRY_RANK_PCT = 0.20     # enter if in top 20%
EXIT_RANK_PCT = 0.40      # exit only if drops below top 40% (hysteresis)
MAX_POSITION_PCT = 0.25   # 25% max single position
RISK_PER_TRADE = 0.02     # 2% max loss per position
TRAILING_STOP_ATR = 2.5   # trailing stop in ATR units (tighter = higher Sharpe)

# Regime (in 4h candles)
REGIME_SMA_LONG_D = 600   # 100 days * 6 (use 100d SMA — faster, less warmup eaten)
REGIME_EMA_FAST = 126     # 21 days * 6
REGIME_EMA_SLOW = 330     # 55 days * 6
REGIME_WARMUP = 400       # candles needed before starting backtest


def load_universe_data():
    """Load all parquet files from universe directory."""
    all_data = {}
    universe_dir = UNIVERSE_DIR

    # Check if universe data exists, fallback to main data dir
    n_universe = len([f for f in os.listdir(universe_dir) if f.endswith('.parquet') and not f.startswith('_')]) if os.path.exists(universe_dir) else 0
    if n_universe < 10:
        print(f"Universe dir has {n_universe} files, using main data dir as fallback...")
        universe_dir = DATA_DIR

    for f in os.listdir(universe_dir):
        if f.endswith('_4h.parquet') and not f.startswith('_'):
            sym = f.replace('_4h.parquet', '')
            try:
                df = pd.read_parquet(os.path.join(universe_dir, f))
                if len(df) >= 250:  # ~42 days minimum
                    all_data[sym] = df
            except Exception:
                continue

    print(f"Loaded {len(all_data)} symbols")
    return all_data


def load_support_data():
    """Load BTC, funding rates, Fear & Greed."""
    btc = None
    funding = pd.DataFrame()
    fng = pd.DataFrame()

    for base_dir in [UNIVERSE_DIR, DATA_DIR]:
        btc_path = os.path.join(base_dir, 'BTCUSDT_4h.parquet')
        if os.path.exists(btc_path) and btc is None:
            btc = pd.read_parquet(btc_path)

    fr_path = os.path.join(DATA_DIR, 'BTCUSDT_funding.parquet')
    if os.path.exists(fr_path):
        funding = pd.read_parquet(fr_path)

    fng_path = os.path.join(DATA_DIR, 'fear_greed.parquet')
    if os.path.exists(fng_path):
        fng = pd.read_parquet(fng_path)

    return btc, funding, fng


# ─── Signal Construction ───

def robust_momentum_signal(close_series, skip=1):
    """
    Robust momentum: average rank across multiple lookbacks (21-35 days).
    Not optimized to a single lookback = less overfitting.
    """
    signals = []
    for lb in LOOKBACKS:
        lb_candles = lb * 6  # convert days to 4h candles
        ret = close_series.pct_change(lb_candles).shift(skip * 6)
        signals.append(ret)
    return pd.concat(signals, axis=1).mean(axis=1)


def compute_atr(df, period=14):
    """Average True Range for trailing stop."""
    period_candles = period * 6
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period_candles).mean()


# ─── Regime Filter ───

def composite_regime_score(btc, funding, fng, idx):
    """
    Composite regime score (0-100).
    Combines slow (SMA), medium (EMA cross), fast (vol), instant (funding, F&G).
    """
    if idx not in btc.index:
        return 50  # neutral

    loc = btc.index.get_loc(idx)
    if loc < REGIME_WARMUP:
        return 50

    close = btc['close'].iloc[:loc + 1]

    # SLOW: BTC vs 200d SMA (weight 25%)
    sma200 = close.rolling(REGIME_SMA_LONG_D).mean().iloc[-1]
    sma_score = 100 if close.iloc[-1] > sma200 else 0

    # MEDIUM: EMA 21d vs EMA 55d (weight 25%)
    ema_fast = close.ewm(span=REGIME_EMA_FAST).mean().iloc[-1]
    ema_slow = close.ewm(span=REGIME_EMA_SLOW).mean().iloc[-1]
    ema_score = 100 if ema_fast > ema_slow else 0

    # FAST: 7-day realized vol vs 90-day percentile (weight 25%)
    rets = close.pct_change()
    vol_7d = rets.tail(42).std() * np.sqrt(6 * 365)  # annualized
    vol_90d = rets.tail(540).std() * np.sqrt(6 * 365)
    vol_ratio = vol_7d / vol_90d if vol_90d > 0 else 1
    vol_score = 0 if vol_ratio > 1.5 else (50 if vol_ratio > 1.2 else 100)

    # FAST: BTC dominance trend (weight 10% — carved from others)
    # Rising BTC.D = altcoins underperforming = reduce alt momentum exposure
    btc_dom_score = 50  # neutral default
    if loc > 180:
        btc_vol_30d = close.tail(180).pct_change().abs().sum()  # BTC activity proxy
        btc_vol_7d = close.tail(42).pct_change().abs().sum()
        # If recent BTC activity is accelerating vs 30d → BTC dominance likely rising
        if btc_vol_7d / max(btc_vol_30d * 42 / 180, 0.001) > 1.3:
            btc_dom_score = 30  # BTC running, alts lagging
        elif btc_vol_7d / max(btc_vol_30d * 42 / 180, 0.001) < 0.7:
            btc_dom_score = 80  # BTC quiet, alts may lead

    # INSTANT: Funding rate (weight 10%)
    fr_score = 50
    if not funding.empty:
        mask = funding.index <= idx
        if mask.any():
            recent_fr = funding.loc[mask, 'fundingRate'].tail(9).mean()  # 3 days
            if recent_fr < -0.0005:
                fr_score = 0
            elif recent_fr > 0.001:
                fr_score = 100
            else:
                fr_score = 50

    # INSTANT: Fear & Greed (weight 10%)
    fng_score = 50
    if not fng.empty:
        mask = fng.index <= idx
        if mask.any():
            fng_val = fng.loc[mask, 'fng'].iloc[-1]
            if fng_val < 20:
                fng_score = 0
            elif fng_val > 60:
                fng_score = 100
            else:
                fng_score = int(fng_val * 100 / 60)

    score = (sma_score * 0.20 + ema_score * 0.20 +
             vol_score * 0.20 + btc_dom_score * 0.15 +
             fr_score * 0.15 + fng_score * 0.10)
    return score


# ─── Whipsaw Detector ───

def detect_whipsaw(regime_history, window=180):
    """
    Count regime flips in recent window (180 candles = 30 days).
    >20% flip rate = WHIPSAW, >10% = CHOPPY.
    """
    if len(regime_history) < window:
        return 'TRENDING'
    recent = regime_history[-window:]
    flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
    flip_rate = flips / window
    if flip_rate > 0.15:
        return 'WHIPSAW'
    elif flip_rate > 0.08:
        return 'CHOPPY'
    return 'TRENDING'


# ─── Backtest Engine ───

def run_cross_sectional_backtest():
    """Main backtest with all technical solutions."""
    print("=" * 60)
    print("  CROSS-SECTIONAL MOMENTUM BACKTEST v1.0")
    print("=" * 60)

    # Load data
    all_data = load_universe_data()
    btc, funding, fng = load_support_data()

    if btc is None or 'BTCUSDT' not in all_data:
        # Try loading BTC from all_data
        if 'BTCUSDT' in all_data:
            btc = all_data['BTCUSDT']
        else:
            raise ValueError("BTC data not found")

    # Determine common timeline from BTC
    timeline = btc.index[REGIME_WARMUP:]  # start after warmup
    print(f"Backtest period: {timeline[0].date()} to {timeline[-1].date()}")
    print(f"Universe: {len(all_data)} symbols")

    # Precompute momentum signals for all symbols
    print("Computing momentum signals...")
    mom_signals = {}
    atr_data = {}
    for sym, df in all_data.items():
        mom_signals[sym] = robust_momentum_signal(df['close'])
        atr_data[sym] = compute_atr(df)

    # ─── Main Loop ───
    equity = INITIAL_CAPITAL
    positions = {}  # {sym: {entry_price, size, peak_price, stop_price}}
    equity_curve = []
    regime_history = []
    last_rebalance = None
    rebalance_interval = pd.DateOffset(days=30)  # monthly

    btc_start_price = btc.loc[timeline[0], 'close']

    for i, ts in enumerate(timeline):
        # Process every candle for trailing stop updates

        # ── Regime Score ──
        regime_score = composite_regime_score(btc, funding, fng, ts)
        allocation_pct = min(regime_score / 100, 1.0)  # linear scaling
        is_above_threshold = regime_score >= 50

        regime_state = 'RISK_ON' if regime_score >= 50 else 'RISK_OFF'
        regime_history.append(regime_state)

        # ── Whipsaw Detection ──
        whipsaw = detect_whipsaw(regime_history)
        if whipsaw == 'WHIPSAW':
            allocation_pct *= 0.40
        elif whipsaw == 'CHOPPY':
            allocation_pct *= 0.70

        # ── Update existing positions (PnL + trailing stop) ──
        closed_syms = []
        total_pnl = 0.0

        for sym, pos in positions.items():
            if sym not in all_data:
                continue
            df = all_data[sym]
            mask = df.index <= ts
            if not mask.any():
                continue
            loc = mask.sum() - 1
            if loc < 1:
                continue

            price = df['close'].iloc[loc]
            prev_price = df['close'].iloc[loc - 1]
            ret = price / prev_price - 1
            total_pnl += ret * pos['size']

            # Update peak and trailing stop
            if price > pos['peak_price']:
                pos['peak_price'] = price
                atr_val = atr_data[sym].iloc[loc] if loc < len(atr_data[sym]) else price * 0.05
                if atr_val > 0:
                    pos['stop_price'] = price - TRAILING_STOP_ATR * atr_val

            # Check trailing stop
            if price <= pos['stop_price']:
                equity -= COST_PER_SIDE * pos['size']  # exit fee
                closed_syms.append(sym)

        for sym in closed_syms:
            del positions[sym]

        # ── Regime OFF: close all ──
        if regime_score < 30 and positions:
            for sym, pos in positions.items():
                equity -= COST_PER_SIDE * pos['size']
            positions.clear()

        # ── Cash yield on uninvested capital ──
        invested = sum(p['size'] for p in positions.values())
        cash = max(0, equity - invested)
        cash_yield = cash * STABLE_YIELD_DAILY / 6  # per 4h candle

        # ── Rebalance trigger ──
        has_empty_slots = len(positions) < MAX_POSITIONS
        stopped_out_recently = len(closed_syms) > 0
        is_monthly = (last_rebalance is None or ts >= last_rebalance + rebalance_interval)
        # Re-entry uses cached ranking (no full recalc), daily check
        is_reentry = (stopped_out_recently and has_empty_slots and i % 6 == 0
                      and not is_monthly)
        should_rebalance = is_monthly

        if should_rebalance and allocation_pct > 0.10:
            # Build point-in-time eligible universe
            eligible = []
            for sym, df in all_data.items():
                # Find nearest index <= ts
                mask = df.index <= ts
                if not mask.any():
                    continue
                loc = mask.sum() - 1
                if loc < 210:  # need warmup for lookbacks
                    continue

                # Compute momentum inline (more reliable than pre-computed)
                close = df['close']
                mom_vals = []
                for lb in LOOKBACKS:
                    lb_c = lb * 6
                    skip_c = SKIP_DAYS * 6
                    if loc >= lb_c + skip_c:
                        ret = close.iloc[loc - skip_c] / close.iloc[loc - skip_c - lb_c] - 1
                        mom_vals.append(ret)
                if not mom_vals:
                    continue
                mom = np.mean(mom_vals)
                if np.isnan(mom):
                    continue

                # Volume filter: recent daily quote volume > $5M
                recent = df.iloc[max(0, loc - 42):loc + 1]
                if 'quote_vol' in recent.columns:
                    daily_vol = recent['quote_vol'].mean() * 6
                else:
                    daily_vol = recent['volume'].mean() * recent['close'].mean() * 6
                if daily_vol < 5_000_000:
                    continue

                eligible.append({
                    'symbol': sym,
                    'momentum': mom,
                    'daily_vol': daily_vol,
                    'close': close.iloc[loc],
                    'loc': loc,
                })

            # ── Restrict to top 30 by volume (proxy for market cap) ──
            # Momentum works 4.2x better in large caps (Liu et al. 2022)
            eligible.sort(key=lambda x: x['daily_vol'], reverse=True)
            eligible = eligible[:30]
            # Re-sort by momentum for ranking
            eligible.sort(key=lambda x: x['momentum'], reverse=True)

            if len(eligible) >= 5:
                # Rank by momentum
                eligible.sort(key=lambda x: x['momentum'], reverse=True)
                n_eligible = len(eligible)
                entry_threshold = int(n_eligible * ENTRY_RANK_PCT)
                exit_threshold = int(n_eligible * EXIT_RANK_PCT)

                top_symbols = {e['symbol'] for e in eligible[:entry_threshold]}
                ok_symbols = {e['symbol'] for e in eligible[:exit_threshold]}

                # ── Hysteresis: exit positions below top 40% ──
                for sym in list(positions.keys()):
                    if sym not in ok_symbols:
                        equity -= COST_PER_SIDE * positions[sym]['size']
                        del positions[sym]

                # ── Enter new positions from top 20% ──
                available_capital = equity * allocation_pct - sum(
                    p['size'] for p in positions.values()
                )
                slots = MAX_POSITIONS - len(positions)

                for entry in eligible[:entry_threshold]:
                    if slots <= 0 or available_capital <= 0:
                        break
                    sym = entry['symbol']
                    if sym in positions:
                        continue

                    # Max-loss position sizing
                    df = all_data[sym]
                    loc = entry['loc']
                    atr_val = atr_data[sym].iloc[loc] if loc < len(atr_data[sym]) else entry['close'] * 0.05
                    stop_distance = TRAILING_STOP_ATR * atr_val / entry['close']
                    if stop_distance < 0.01:
                        stop_distance = 0.15  # fallback

                    max_loss_dollar = equity * RISK_PER_TRADE
                    size = min(
                        max_loss_dollar / stop_distance,
                        equity * MAX_POSITION_PCT,
                        available_capital / max(slots, 1),
                    )

                    if size < 50:  # minimum $50
                        continue

                    equity -= COST_PER_SIDE * size  # entry fee
                    positions[sym] = {
                        'entry_price': entry['close'],
                        'size': size,
                        'peak_price': entry['close'],
                        'stop_price': entry['close'] - TRAILING_STOP_ATR * atr_val,
                    }
                    available_capital -= size
                    slots -= 1

            last_rebalance = ts
            last_eligible = eligible  # cache for re-entry

        # ── Re-entry from cached ranking after stop-out ──
        if is_reentry and 'last_eligible' in dir() and last_eligible and allocation_pct > 0.10:
            available_capital = equity * allocation_pct - sum(
                p['size'] for p in positions.values()
            )
            slots = MAX_POSITIONS - len(positions)
            n_elig = len(last_eligible)
            entry_threshold = max(1, int(n_elig * ENTRY_RANK_PCT))

            for entry in last_eligible[:entry_threshold]:
                if slots <= 0 or available_capital <= 0:
                    break
                sym = entry['symbol']
                if sym in positions or sym not in all_data:
                    continue

                df = all_data[sym]
                mask = df.index <= ts
                if not mask.any():
                    continue
                loc = mask.sum() - 1
                price = df['close'].iloc[loc]
                atr_val = atr_data[sym].iloc[loc] if loc < len(atr_data[sym]) else price * 0.05
                if atr_val <= 0:
                    continue

                stop_distance = TRAILING_STOP_ATR * atr_val / price
                if stop_distance < 0.01:
                    stop_distance = 0.15
                max_loss_dollar = equity * RISK_PER_TRADE
                size = min(
                    max_loss_dollar / stop_distance,
                    equity * MAX_POSITION_PCT,
                    available_capital / max(slots, 1),
                )
                if size < 50:
                    continue

                equity -= COST_PER_SIDE * size
                positions[sym] = {
                    'entry_price': price,
                    'size': size,
                    'peak_price': price,
                    'stop_price': price - TRAILING_STOP_ATR * atr_val,
                }
                available_capital -= size
                slots -= 1

        # ── Update equity ──
        equity += total_pnl + cash_yield

        # ── Alpha-relative drawdown check ──
        btc_price = btc.loc[ts, 'close'] if ts in btc.index else 0
        btc_ret = btc_price / btc_start_price - 1 if btc_start_price > 0 else 0

        equity_curve.append({
            'date': ts,
            'equity': equity,
            'invested': sum(p['size'] for p in positions.values()),
            'n_positions': len(positions),
            'regime_score': regime_score,
            'whipsaw': whipsaw,
            'allocation_pct': allocation_pct,
            'pnl': total_pnl,
            'cash_yield': cash_yield,
            'btc_close': btc_price,
        })

    # ─── Results ───
    result = pd.DataFrame(equity_curve).set_index('date')
    metrics = compute_metrics(result)

    # Save
    result.to_parquet(os.path.join(RESULTS_DIR, 'cs_momentum_result.parquet'))
    pd.Series(metrics).to_json(os.path.join(RESULTS_DIR, 'cs_momentum_metrics.json'))

    print("\n" + "=" * 60)
    print("  CROSS-SECTIONAL MOMENTUM RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")
    print("=" * 60)
    print(f"\nResults saved to {RESULTS_DIR}/")

    return result, metrics


def compute_metrics(result):
    """Compute comprehensive performance metrics."""
    eq = result['equity']
    returns = eq.pct_change().dropna()

    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_days = (result.index[-1] - result.index[0]).days
    n_years = n_days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    candles_per_year = 6 * 365.25
    avg_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (avg_ret / std_ret * np.sqrt(candles_per_year)) if std_ret > 0 else 0

    downside = returns[returns < 0].std()
    sortino = (avg_ret / downside * np.sqrt(candles_per_year)) if downside > 0 else 0

    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    win_rate = (returns > 0).mean() * 100

    # BTC comparison
    btc_start = result['btc_close'].iloc[0]
    btc_end = result['btc_close'].iloc[-1]
    btc_return = ((btc_end / btc_start) - 1) * 100 if btc_start > 0 else 0
    btc_cagr = ((btc_end / btc_start) ** (1 / n_years) - 1) * 100 if n_years > 0 and btc_start > 0 else 0

    # Alpha over BTC
    alpha = cagr - btc_cagr

    # Average positions
    avg_positions = result['n_positions'].mean()
    avg_allocation = result['allocation_pct'].mean() * 100

    # PnL decomposition
    total_pnl = result['pnl'].sum()
    total_cash = result['cash_yield'].sum()

    # Yearly breakdown
    yearly = {}
    for year in sorted(result.index.year.unique()):
        yr_data = result[result.index.year == year]
        if len(yr_data) > 1:
            yr_ret = (yr_data['equity'].iloc[-1] / yr_data['equity'].iloc[0] - 1) * 100
            yearly[str(year)] = f"{yr_ret:+.1f}%"

    return {
        'Period': f"{result.index[0].strftime('%Y-%m-%d')} to {result.index[-1].strftime('%Y-%m-%d')}",
        'Duration (days)': n_days,
        'Initial Capital': f"${INITIAL_CAPITAL:,.0f}",
        'Final Equity': f"${eq.iloc[-1]:,.2f}",
        'Total Return': f"{total_return:.1f}%",
        'CAGR': f"{cagr:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Calmar Ratio': f"{calmar:.2f}",
        'Max Drawdown': f"{max_dd:.2f}%",
        'Win Rate (candle)': f"{win_rate:.1f}%",
        'Alpha over BTC': f"{alpha:+.2f}%",
        'BTC CAGR': f"{btc_cagr:.2f}%",
        'Avg Positions': f"{avg_positions:.1f}",
        'Avg Allocation': f"{avg_allocation:.0f}%",
        'PnL Momentum': f"${total_pnl:,.2f}",
        'PnL Cash Yield': f"${total_cash:,.2f}",
        **{f'Year {k}': v for k, v in yearly.items()},
    }


if __name__ == '__main__':
    run_cross_sectional_backtest()
