"""
Strategy #3 — Statistical Arbitrage / Pairs Trading
Cointegrated pairs with z-score entry/exit.
COST-AWARE: minimal turnover, only majors, wide z-score thresholds.
"""
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.stattools import coint

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

MAKER_FEE = 0.0002
SLIPPAGE = 0.0005
COST_PER_SIDE = MAKER_FEE + SLIPPAGE
FUNDING_COST_SHORT = 0.0001 / 6  # per 4h candle for short leg

# ONLY major pairs — liquid, low slippage, low borrow cost
PAIRS = [
    ('BTCUSDT', 'ETHUSDT'),
    ('ETHUSDT', 'SOLUSDT'),
    ('ETHUSDT', 'BNBUSDT'),
]

# Conservative thresholds to minimize turnover
Z_ENTRY = 2.0       # enter when z-score > 2.0 (wide)
Z_EXIT = 0.3        # exit near mean
Z_STOP = 4.0        # stop loss at 4.0 sigma
LOOKBACK = 180       # 180 candles of 4h = 30 days for spread estimation
COINT_WINDOW = 540   # 540 candles = 90 days for cointegration test
RETEST_EVERY = 180   # retest cointegration every 30 days


def load_pair_data(sym_a: str, sym_b: str) -> tuple:
    path_a = os.path.join(DATA_DIR, f'{sym_a}_4h.parquet')
    path_b = os.path.join(DATA_DIR, f'{sym_b}_4h.parquet')
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return None, None
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    common = df_a.index.intersection(df_b.index)
    return df_a.loc[common], df_b.loc[common]


def test_cointegration(prices_a: pd.Series, prices_b: pd.Series) -> tuple:
    """Returns (is_cointegrated, p_value, hedge_ratio)."""
    try:
        score, pvalue, _ = coint(np.log(prices_a), np.log(prices_b))
        # Simple OLS hedge ratio
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)
        hedge_ratio = np.polyfit(log_b, log_a, 1)[0]
        return pvalue < 0.05, pvalue, hedge_ratio
    except Exception:
        return False, 1.0, 1.0


def run_pairs_backtest(initial_capital: float = 10000) -> pd.DataFrame:
    """
    Pairs trading with cointegration.
    Cost-conscious: wide z-score thresholds, infrequent rebalance.
    """
    # Load BTC for benchmark
    btc = pd.read_parquet(os.path.join(DATA_DIR, 'BTCUSDT_4h.parquet'))

    all_results = []
    capital_per_pair = initial_capital / len(PAIRS)

    for sym_a, sym_b in PAIRS:
        df_a, df_b = load_pair_data(sym_a, sym_b)
        if df_a is None:
            continue

        equity = capital_per_pair
        position = 0  # 1 = long spread, -1 = short spread, 0 = flat
        entry_z = 0.0
        is_coint = False
        hedge_ratio = 1.0
        trades = 0

        for i in range(COINT_WINDOW, len(df_a)):
            ts = df_a.index[i]

            # Retest cointegration periodically
            if i % RETEST_EVERY == 0:
                window_a = df_a['close'].iloc[i-COINT_WINDOW:i]
                window_b = df_b['close'].iloc[i-COINT_WINDOW:i]
                is_coint, pval, hedge_ratio = test_cointegration(window_a, window_b)

            if not is_coint:
                # Not cointegrated — stay flat
                if position != 0:
                    equity -= abs(position) * 2 * COST_PER_SIDE * equity  # exit cost
                    position = 0
                    trades += 1
                all_results.append({
                    'date': ts, 'pair': f'{sym_a}/{sym_b}',
                    'equity': equity, 'pnl': 0, 'position': 0,
                    'z_score': 0, 'trades': trades
                })
                continue

            # Calculate spread and z-score
            log_a = np.log(df_a['close'].iloc[i-LOOKBACK:i+1])
            log_b = np.log(df_b['close'].iloc[i-LOOKBACK:i+1])
            spread = log_a.values - hedge_ratio * log_b.values
            mean_spread = spread[:-1].mean()
            std_spread = spread[:-1].std()

            if std_spread < 1e-10:
                all_results.append({
                    'date': ts, 'pair': f'{sym_a}/{sym_b}',
                    'equity': equity, 'pnl': 0, 'position': position,
                    'z_score': 0, 'trades': trades
                })
                continue

            z = (spread[-1] - mean_spread) / std_spread

            # Position PnL (if holding)
            pnl = 0.0
            if position != 0 and i > 0:
                ret_a = df_a['close'].iloc[i] / df_a['close'].iloc[i-1] - 1
                ret_b = df_b['close'].iloc[i] / df_b['close'].iloc[i-1] - 1
                # Long spread = long A, short B (scaled by hedge ratio)
                spread_ret = ret_a - hedge_ratio * ret_b
                pnl = position * spread_ret * equity * 0.5  # use 50% of equity
                # Funding cost on short leg
                pnl -= abs(position) * equity * 0.5 * FUNDING_COST_SHORT

            equity += pnl

            # Entry/exit logic
            if position == 0:
                if z > Z_ENTRY:
                    position = -1  # short spread (expect mean reversion down)
                    equity -= 2 * COST_PER_SIDE * equity * 0.5  # entry cost both legs
                    trades += 1
                elif z < -Z_ENTRY:
                    position = 1   # long spread (expect mean reversion up)
                    equity -= 2 * COST_PER_SIDE * equity * 0.5
                    trades += 1
            elif position != 0:
                # Exit conditions
                if abs(z) < Z_EXIT:  # mean reverted
                    equity -= 2 * COST_PER_SIDE * equity * 0.5  # exit cost
                    position = 0
                    trades += 1
                elif abs(z) > Z_STOP:  # stop loss
                    equity -= 2 * COST_PER_SIDE * equity * 0.5
                    position = 0
                    trades += 1

            all_results.append({
                'date': ts, 'pair': f'{sym_a}/{sym_b}',
                'equity': equity, 'pnl': pnl, 'position': position,
                'z_score': z, 'trades': trades
            })

    if not all_results:
        return pd.DataFrame()

    # Aggregate across pairs
    df_all = pd.DataFrame(all_results)
    # Sum equity across pairs per timestamp
    agg = df_all.groupby('date').agg({
        'equity': 'sum',
        'pnl': 'sum',
        'trades': 'sum',
    })

    # Add BTC for benchmark
    agg = agg.join(btc[['close']].rename(columns={'close': 'btc_close'}), how='left')
    agg['btc_close'] = agg['btc_close'].ffill()

    return agg


if __name__ == '__main__':
    result = run_pairs_backtest()
    result.to_parquet(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   'results', 'pairs_result.parquet'))
    eq = result['equity']
    n_years = (result.index[-1] - result.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1/n_years) - 1) * 100
    returns = eq.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(6 * 365.25) if returns.std() > 0 else 0
    peak = eq.expanding().max()
    max_dd = ((eq - peak) / peak).min() * 100
    total_trades = result['trades'].iloc[-1]
    print(f"Pairs: ${eq.iloc[0]:,.0f} -> ${eq.iloc[-1]:,.0f}")
    print(f"CAGR: {cagr:.2f}%, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}%")
    print(f"Total trades: {total_trades:.0f}")
