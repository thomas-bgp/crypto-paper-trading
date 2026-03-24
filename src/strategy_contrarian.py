"""
Strategy #2 — Contrarian Long-Only (Mean Reversion)
Buy oversold assets (RSI < 30) from top liquid cryptos.
Market-neutral via position sizing: only invested when oversold signals fire.
In trending markets, stays mostly in cash = low correlation with momentum.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

MAKER_FEE = 0.0002
SLIPPAGE = 0.0005
COST_PER_SIDE = MAKER_FEE + SLIPPAGE

# Only liquid, major assets
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
           'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT']

RSI_BUY = 28       # enter when RSI < 28 (deeply oversold)
RSI_SELL = 55      # exit when RSI > 55 (recovered)
MAX_POSITIONS = 3  # max 3 concurrent positions
POS_SIZE = 0.20    # 20% of equity per position


def run_contrarian_backtest(initial_capital: float = 10000) -> pd.DataFrame:
    """Long-only contrarian: buy RSI oversold, sell on recovery."""
    dfs = {}
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f'{sym}_4h.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['rsi'] = ta.rsi(df['close'], length=14)
            dfs[sym] = df

    btc = dfs.get('BTCUSDT')
    if btc is None:
        raise ValueError("BTC data not found")

    timeline = btc.index
    equity = initial_capital
    positions = {}  # {sym: {'entry_price': x, 'size': x}}
    equity_curve = []

    for i in range(20, len(timeline)):
        ts = timeline[i]
        total_pnl = 0.0

        # Update existing positions PnL
        closed = []
        for sym, pos in positions.items():
            if sym not in dfs or ts not in dfs[sym].index:
                continue
            idx = dfs[sym].index.get_loc(ts)
            if idx < 1:
                continue
            price = dfs[sym]['close'].iloc[idx]
            prev_price = dfs[sym]['close'].iloc[idx - 1]
            ret = price / prev_price - 1
            total_pnl += ret * pos['size']

            # Exit if RSI recovered
            rsi_val = dfs[sym]['rsi'].iloc[idx]
            if not np.isnan(rsi_val) and rsi_val > RSI_SELL:
                # Close position
                equity -= COST_PER_SIDE * pos['size']  # exit cost
                closed.append(sym)

        for sym in closed:
            del positions[sym]

        # Look for new entries (only if we have room)
        if len(positions) < MAX_POSITIONS:
            candidates = []
            for sym, df in dfs.items():
                if sym in positions or ts not in df.index:
                    continue
                idx = df.index.get_loc(ts)
                if idx < 1:
                    continue
                rsi_val = df['rsi'].iloc[idx]
                if not np.isnan(rsi_val) and rsi_val < RSI_BUY:
                    candidates.append((sym, rsi_val))

            # Sort by most oversold first
            candidates.sort(key=lambda x: x[1])
            slots = MAX_POSITIONS - len(positions)

            for sym, rsi_val in candidates[:slots]:
                size = equity * POS_SIZE
                cost = COST_PER_SIDE * size
                equity -= cost
                idx = dfs[sym].index.get_loc(ts)
                positions[sym] = {
                    'entry_price': dfs[sym]['close'].iloc[idx],
                    'size': size
                }

        equity += total_pnl

        equity_curve.append({
            'date': ts,
            'equity': equity,
            'pnl': total_pnl,
            'n_positions': len(positions),
            'btc_close': btc.loc[ts, 'close'] if ts in btc.index else np.nan,
        })

    return pd.DataFrame(equity_curve).set_index('date')


if __name__ == '__main__':
    result = run_contrarian_backtest()
    result.to_parquet(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   'results', 'contrarian_result.parquet'))
    eq = result['equity']
    n_years = (result.index[-1] - result.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1/n_years) - 1) * 100
    returns = eq.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(6 * 365.25) if returns.std() > 0 else 0
    peak = eq.expanding().max()
    max_dd = ((eq - peak) / peak).min() * 100
    print(f"Contrarian: ${eq.iloc[0]:,.0f} -> ${eq.iloc[-1]:,.0f}")
    print(f"CAGR: {cagr:.2f}%, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}%")
    # 2025 specific
    r25 = result[result.index.year == 2025]
    if len(r25) > 1:
        ret25 = (r25['equity'].iloc[-1] / r25['equity'].iloc[0] - 1) * 100
        print(f"2025: {ret25:+.2f}%")
