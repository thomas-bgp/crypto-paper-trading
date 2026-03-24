"""
Event-driven backtester with regime switch.
- Layer 1: Cross-sectional momentum (BULL regime)
- Layer 2: Mean reversion BTC/ETH (SIDEWAYS regime)
- Layer 3: Cash (BEAR regime) — earns stablecoin yield ~5% aa
"""
import numpy as np
import pandas as pd
import os
from regime_detector import rolling_regime_detection
from strategies import momentum_signal, mean_reversion_signal, rank_momentum

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Configuration ───
INITIAL_CAPITAL = 10_000
MAKER_FEE = 0.0002       # 0.02% maker with BNB
TAKER_FEE = 0.00075      # 0.075% taker
SLIPPAGE = 0.0005         # 0.05% estimated slippage
TOTAL_COST = MAKER_FEE + SLIPPAGE  # per side
STABLE_YIELD_DAILY = 0.05 / 365    # 5% annual stablecoin yield
REBALANCE_EVERY = 6       # candles (24h in 4h)

# Capital allocation by regime
ALLOC = {
    'BULL':     {'momentum': 0.80, 'meanrev': 0.00, 'cash': 0.20},
    'SIDEWAYS': {'momentum': 0.30, 'meanrev': 0.00, 'cash': 0.70},
    'BEAR':     {'momentum': 0.00, 'meanrev': 0.00, 'cash': 1.00},
}

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'LTCUSDT', 'ATOMUSDT', 'NEARUSDT', 'ARBUSDT',
    'OPUSDT', 'APTUSDT', 'SUIUSDT', 'INJUSDT', 'TIAUSDT'
]


def load_data():
    """Load all cached data."""
    dfs = {}
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f'{sym}_4h.parquet')
        if os.path.exists(path):
            dfs[sym] = pd.read_parquet(path)

    df_funding = pd.DataFrame()
    fr_path = os.path.join(DATA_DIR, 'BTCUSDT_funding.parquet')
    if os.path.exists(fr_path):
        df_funding = pd.read_parquet(fr_path)

    df_fng = pd.DataFrame()
    fng_path = os.path.join(DATA_DIR, 'fear_greed.parquet')
    if os.path.exists(fng_path):
        df_fng = pd.read_parquet(fng_path)

    return dfs, df_funding, df_fng


def run_backtest():
    """Main backtest loop."""
    print("Loading data...")
    dfs, df_funding, df_fng = load_data()

    if 'BTCUSDT' not in dfs:
        raise ValueError("BTC data not found. Run data_fetcher.py first.")

    btc = dfs['BTCUSDT']
    print(f"BTC data: {btc.index[0]} to {btc.index[-1]} ({len(btc)} candles)")

    # ─── Regime Detection (walk-forward, no lookahead) ───
    print("Running regime detection (walk-forward)...")
    regimes = rolling_regime_detection(btc, df_funding, df_fng,
                                       refit_every=42, lookback=500)
    print(f"Regimes detected: {regimes['regime'].value_counts().to_dict()}")

    # ─── Compute momentum signals for all symbols ───
    print("Computing momentum signals...")
    mom_signals = {}
    for sym, df in dfs.items():
        mom_signals[sym] = momentum_signal(df, fast=20, slow=50)

    # ─── Compute mean reversion signal (BTC only for simplicity) ───
    print("Computing mean reversion signals...")
    mr_signal = mean_reversion_signal(btc)

    # ─── Align to BTC + regimes index (don't require ALL symbols) ───
    common_idx = btc.index.intersection(regimes.index).sort_values()
    print(f"Backtest period: {common_idx[0]} to {common_idx[-1]} ({len(common_idx)} candles)")

    # ─── Backtest Loop ───
    equity = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    current_positions = {}  # {symbol: weight}
    last_rebalance = 0

    prev_mom_weight = 0.30  # track for turnover cost

    for i, ts in enumerate(common_idx):
        regime = regimes.loc[ts, 'regime'] if ts in regimes.index else 'SIDEWAYS'

        # ── Fix 3: Continuous allocation from regime detector ──
        if ts in regimes.index and 'mom_weight' in regimes.columns:
            mom_weight = float(regimes.loc[ts, 'mom_weight'])
        else:
            mom_weight = 0.30  # default SIDEWAYS-ish
        cash_weight = 1.0 - mom_weight

        # ── Cash yield (daily) ──
        cash_yield = equity * cash_weight * STABLE_YIELD_DAILY / 6

        # ── Momentum layer (rebalance periodically) ──
        mom_pnl = 0.0
        if i % REBALANCE_EVERY == 0 and mom_weight > 0.05:
            ret_dict = {}
            for sym, df in dfs.items():
                if ts in df.index:
                    ret_dict[sym] = df.loc[:ts]
            new_weights = rank_momentum(ret_dict, lookback=24)

            # Turnover cost (proportional to allocation change)
            old_syms = set(current_positions.keys())
            new_syms = set(new_weights.keys())
            turnover = len(old_syms.symmetric_difference(new_syms))
            alloc_change = abs(mom_weight - prev_mom_weight)
            rebalance_cost = (turnover * TOTAL_COST * equity * mom_weight / max(len(new_weights), 1)
                              + alloc_change * TOTAL_COST * equity)

            equity -= rebalance_cost
            current_positions = new_weights
            prev_mom_weight = mom_weight

        # Calculate momentum PnL
        if current_positions and mom_weight > 0.05:
            for sym, weight in current_positions.items():
                if sym not in dfs:
                    continue
                sym_df = dfs[sym]
                if ts not in sym_df.index:
                    continue
                idx = sym_df.index.get_loc(ts)
                if idx > 0:
                    ret = sym_df['close'].iloc[idx] / sym_df['close'].iloc[idx - 1] - 1
                    mom_pnl += ret * weight * equity * mom_weight

        # ── Mean reversion removed (was destroying value) ──
        mr_pnl = 0.0

        # ── Update equity ──
        total_pnl = mom_pnl + mr_pnl + cash_yield
        equity += total_pnl

        equity_curve.append({
            'date': ts,
            'equity': equity,
            'regime': regime,
            'mom_pnl': mom_pnl,
            'mr_pnl': mr_pnl,
            'cash_yield': cash_yield,
            'total_pnl': total_pnl,
            'btc_close': btc.loc[ts, 'close'] if ts in btc.index else np.nan,
            'confidence': regimes.loc[ts, 'confidence'] if ts in regimes.index else 0,
            'funding_rate': regimes.loc[ts, 'fundingRate'] if ts in regimes.index else 0,
            'fng': regimes.loc[ts, 'fng'] if ts in regimes.index else 50,
        })

    result = pd.DataFrame(equity_curve).set_index('date')

    # ─── Metrics ───
    metrics = compute_metrics(result)

    # ─── Save ───
    result.to_parquet(os.path.join(RESULTS_DIR, 'backtest_result.parquet'))
    regimes.to_parquet(os.path.join(RESULTS_DIR, 'regimes.parquet'))
    pd.Series(metrics).to_json(os.path.join(RESULTS_DIR, 'metrics.json'))

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")
    print("=" * 60)
    print(f"\nResults saved to {RESULTS_DIR}/")

    return result, regimes, metrics


def compute_metrics(result: pd.DataFrame) -> dict:
    """Compute performance metrics."""
    eq = result['equity']
    returns = eq.pct_change().dropna()

    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_days = (result.index[-1] - result.index[0]).days
    n_years = n_days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Annualized Sharpe (assuming 6 candles/day)
    candles_per_year = 6 * 365.25
    avg_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (avg_ret / std_ret * np.sqrt(candles_per_year)) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0].std()
    sortino = (avg_ret / downside * np.sqrt(candles_per_year)) if downside > 0 else 0

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (by candle)
    win_rate = (returns > 0).mean() * 100

    # PnL decomposition
    total_mom = result['mom_pnl'].sum()
    total_mr = result['mr_pnl'].sum()
    total_cash = result['cash_yield'].sum()
    total_pnl = result['total_pnl'].sum()

    # Regime distribution
    regime_counts = result['regime'].value_counts(normalize=True) * 100

    # BTC buy & hold comparison
    btc_start = result['btc_close'].iloc[0]
    btc_end = result['btc_close'].iloc[-1]
    btc_return = ((btc_end / btc_start) - 1) * 100 if btc_start > 0 else 0
    btc_cagr = ((btc_end / btc_start) ** (1 / n_years) - 1) * 100 if n_years > 0 and btc_start > 0 else 0

    return {
        'Period': f"{result.index[0].strftime('%Y-%m-%d')} to {result.index[-1].strftime('%Y-%m-%d')}",
        'Duration (days)': n_days,
        'Initial Capital': f"${INITIAL_CAPITAL:,.0f}",
        'Final Equity': f"${eq.iloc[-1]:,.2f}",
        'Total Return': f"{total_return:.2f}%",
        'CAGR': f"{cagr:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Calmar Ratio': f"{calmar:.2f}",
        'Max Drawdown': f"{max_dd:.2f}%",
        'Win Rate (candle)': f"{win_rate:.1f}%",
        'PnL Momentum': f"${total_mom:,.2f}",
        'PnL Mean Rev': f"${total_mr:,.2f}",
        'PnL Cash Yield': f"${total_cash:,.2f}",
        'BTC Buy&Hold Return': f"{btc_return:.2f}%",
        'BTC CAGR': f"{btc_cagr:.2f}%",
        'Regime BULL %': f"{regime_counts.get('BULL', 0):.1f}%",
        'Regime SIDEWAYS %': f"{regime_counts.get('SIDEWAYS', 0):.1f}%",
        'Regime BEAR %': f"{regime_counts.get('BEAR', 0):.1f}%",
    }


if __name__ == '__main__':
    run_backtest()
