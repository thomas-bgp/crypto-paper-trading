"""
Trading strategies — 3 layers with regime switch.
Layer 1: Cross-sectional Momentum (BULL)
Layer 2: Mean Reversion (SIDEWAYS)
Layer 3: Cash / Stablecoin yield (BEAR)
"""
import numpy as np
import pandas as pd
import pandas_ta as ta


def momentum_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Donchian Channel breakout signal.
    Returns: 1 (long), 0 (flat), -1 (short)
    """
    high_fast = df['high'].rolling(fast).max()
    low_fast = df['low'].rolling(fast).min()
    high_slow = df['high'].rolling(slow).max()
    low_slow = df['low'].rolling(slow).min()

    signal = pd.Series(0, index=df.index)
    # Long: close breaks above fast channel AND above slow channel
    signal = np.where(
        (df['close'] > high_fast.shift(1)) & (df['close'] > high_slow.shift(1)), 1,
        np.where(
            (df['close'] < low_fast.shift(1)) & (df['close'] < low_slow.shift(1)), -1, 0
        )
    )
    return pd.Series(signal, index=df.index).ffill()


def mean_reversion_signal(df: pd.DataFrame, rsi_period: int = 14,
                          bb_period: int = 20, bb_std: float = 2.0) -> pd.Series:
    """
    RSI + Bollinger Band mean reversion.
    Only active when ADX < 25 (no strong trend).
    """
    rsi = ta.rsi(df['close'], length=rsi_period)
    bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
    # Find actual column names dynamically
    bbl_col = [c for c in bb.columns if c.startswith('BBL')][0]
    bbu_col = [c for c in bb.columns if c.startswith('BBU')][0]
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    adx = adx_df['ADX_14']

    signal = pd.Series(0.0, index=df.index)

    # Only trade when ADX < 25 (sideways market)
    sideways = adx < 25

    # Buy when RSI oversold AND price at lower Bollinger
    buy = sideways & (rsi < 30) & (df['close'] < bb[bbl_col])
    # Sell when RSI overbought AND price at upper Bollinger
    sell = sideways & (rsi > 70) & (df['close'] > bb[bbu_col])
    # Exit when RSI returns to neutral
    exit_long = rsi > 50
    exit_short = rsi < 50

    pos = 0.0
    positions = []
    for i in range(len(df)):
        if buy.iloc[i] and pos <= 0:
            pos = 1.0
        elif sell.iloc[i] and pos >= 0:
            pos = -1.0
        elif pos > 0 and exit_long.iloc[i]:
            pos = 0.0
        elif pos < 0 and exit_short.iloc[i]:
            pos = 0.0
        positions.append(pos)

    return pd.Series(positions, index=df.index)


def rank_momentum(returns_dict: dict, lookback: int = 24) -> dict:
    """
    Cross-sectional momentum ranking.
    Returns allocation weights: top quintile gets positive weight, bottom gets 0.
    lookback: number of 4h candles (24 = 4 days)
    """
    scores = {}
    for sym, df in returns_dict.items():
        if len(df) < lookback:
            continue
        ret = df['close'].pct_change(lookback).iloc[-1]
        if not np.isnan(ret):
            scores[sym] = ret

    if not scores:
        return {}

    sorted_syms = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
    n = len(sorted_syms)
    top_n = max(1, n // 5)  # top 20%

    weights = {}
    for sym in sorted_syms[:top_n]:
        weights[sym] = 1.0 / top_n
    return weights
