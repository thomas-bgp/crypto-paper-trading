"""
Vectorized Feature Engine — computes all factors for all coins at once.
Returns a panel DataFrame: (date, symbol) → features
"""
import numpy as np
import pandas as pd
import os
from functools import lru_cache

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
UNIVERSE_DIR = os.path.join(DATA_DIR, 'universe')


def load_all_data(min_candles=250):
    """Load all parquet files into a dict of DataFrames."""
    data = {}
    for base_dir in [UNIVERSE_DIR, DATA_DIR]:
        if not os.path.exists(base_dir):
            continue
        for f in os.listdir(base_dir):
            if f.endswith('_4h.parquet') and not f.startswith('_'):
                sym = f.replace('_4h.parquet', '')
                if sym in data:
                    continue
                try:
                    df = pd.read_parquet(os.path.join(base_dir, f))
                    if len(df) >= min_candles:
                        data[sym] = df
                except Exception:
                    continue
    return data


def build_panel(data_dict, resample='1D'):
    """
    Build a panel DataFrame from dict of DataFrames.
    Resamples to daily (or keeps 4h) for speed.
    Returns MultiIndex DataFrame with (date, symbol) index.
    """
    frames = []
    for sym, df in data_dict.items():
        if resample and resample != '4h':
            # Resample 4h → daily using last close, max high, min low, sum volume
            daily = df.resample(resample).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna(subset=['close'])
            if 'quote_vol' in df.columns:
                qv = df['quote_vol'].resample(resample).sum()
                daily['quote_vol'] = qv
            daily['symbol'] = sym
            frames.append(daily)
        else:
            df = df.copy()
            df['symbol'] = sym
            frames.append(df)

    panel = pd.concat(frames)
    panel = panel.reset_index().rename(columns={'open_time': 'date', 'index': 'date'})
    if 'date' not in panel.columns:
        panel = panel.reset_index().rename(columns={panel.index.name or 'index': 'date'})
    panel = panel.set_index(['date', 'symbol']).sort_index()
    return panel


def compute_features(panel, lookbacks=None):
    """
    Compute ALL features vectorized. Input: panel with (date, symbol) index.
    Returns: same panel with feature columns added.
    """
    if lookbacks is None:
        lookbacks = [7, 14, 21, 28, 35, 56, 90]

    # Work per symbol (groupby symbol, apply vectorized ops)
    grouped = panel.groupby(level='symbol')

    # ─── 1. MOMENTUM FEATURES (returns at various lookbacks) ───
    for lb in lookbacks:
        panel[f'ret_{lb}d'] = grouped['close'].pct_change(lb)

    # Skip-1 momentum (avoid daily reversal)
    for lb in [21, 28, 35]:
        panel[f'ret_{lb}d_skip1'] = grouped['close'].shift(1).transform(
            lambda x: x.pct_change(lb)
        )

    # ─── 2. VOLATILITY FEATURES ───
    log_ret = grouped['close'].transform(lambda x: np.log(x / x.shift(1)))
    panel['log_ret'] = log_ret

    for window in [14, 28, 56]:
        panel[f'rvol_{window}d'] = grouped['log_ret'].transform(
            lambda x: x.rolling(window).std() * np.sqrt(365)
        )

    # Vol of vol
    panel['vol_of_vol'] = grouped['rvol_28d'].transform(
        lambda x: x.rolling(28).std()
    )

    # ─── 3. HIGHER MOMENTS ───
    panel['skew_28d'] = grouped['log_ret'].transform(
        lambda x: x.rolling(28).skew()
    )
    panel['kurt_28d'] = grouped['log_ret'].transform(
        lambda x: x.rolling(28).kurt()
    )

    # Max / Min daily return
    panel['max_ret_28d'] = grouped['log_ret'].transform(
        lambda x: x.rolling(28).max()
    )
    panel['min_ret_28d'] = grouped['log_ret'].transform(
        lambda x: x.rolling(28).min()
    )

    # ─── 4. VOLUME FEATURES ───
    panel['vol_ratio_7d'] = grouped['volume'].transform(
        lambda x: x / x.rolling(28).mean()
    ).rolling(7).mean()

    panel['vol_trend_28d'] = grouped['volume'].transform(
        lambda x: x.rolling(28).apply(
            lambda v: np.polyfit(np.arange(len(v)), np.log(v + 1), 1)[0]
            if len(v) == 28 else np.nan, raw=False
        )
    )

    # Turnover proxy (volume / close as proxy for vol/mcap)
    panel['turnover_28d'] = grouped.apply(
        lambda g: (g['volume'] / g['close']).rolling(28).mean()
    ).droplevel(0).sort_index()

    # ─── 5. TECHNICAL INDICATORS (vectorized) ───

    # RSI
    for period in [14, 28]:
        delta = grouped['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.groupby(level='symbol').transform(
            lambda x: x.ewm(span=period, adjust=False).mean()
        )
        avg_loss = loss.groupby(level='symbol').transform(
            lambda x: x.ewm(span=period, adjust=False).mean()
        )
        rs = avg_gain / (avg_loss + 1e-10)
        panel[f'rsi_{period}d'] = 100 - 100 / (1 + rs)

    # Bollinger %B
    sma20 = grouped['close'].transform(lambda x: x.rolling(20).mean())
    std20 = grouped['close'].transform(lambda x: x.rolling(20).std())
    panel['bb_pctb'] = (panel['close'] - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)
    panel['bb_width'] = 4 * std20 / (sma20 + 1e-10)

    # MACD
    ema12 = grouped['close'].transform(lambda x: x.ewm(span=12).mean())
    ema26 = grouped['close'].transform(lambda x: x.ewm(span=26).mean())
    macd_line = ema12 - ema26
    macd_signal = macd_line.groupby(level='symbol').transform(
        lambda x: x.ewm(span=9).mean()
    )
    panel['macd_hist'] = macd_line - macd_signal
    panel['macd_ratio'] = macd_line / (panel['close'] + 1e-10)

    # Stochastic %K
    low14 = grouped['low'].transform(lambda x: x.rolling(14).min())
    high14 = grouped['high'].transform(lambda x: x.rolling(14).max())
    panel['stoch_k'] = (panel['close'] - low14) / (high14 - low14 + 1e-10) * 100

    # CCI
    tp = (panel['high'] + panel['low'] + panel['close']) / 3
    sma_tp = tp.groupby(level='symbol').transform(lambda x: x.rolling(20).mean())
    mad_tp = tp.groupby(level='symbol').transform(
        lambda x: x.rolling(20).apply(lambda v: np.abs(v - v.mean()).mean(), raw=True)
    )
    panel['cci_20d'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-10)

    # Donchian position
    high20 = grouped['high'].transform(lambda x: x.rolling(20).max())
    low20 = grouped['low'].transform(lambda x: x.rolling(20).min())
    panel['donchian_pos'] = (panel['close'] - low20) / (high20 - low20 + 1e-10)

    # ATR ratio
    tr = pd.concat([
        panel['high'] - panel['low'],
        (panel['high'] - grouped['close'].shift(1)).abs(),
        (panel['low'] - grouped['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    panel['atr_14d'] = tr.groupby(level='symbol').transform(
        lambda x: x.rolling(14).mean()
    )
    panel['atr_ratio'] = panel['atr_14d'] / (panel['close'] + 1e-10)

    # ─── 6. POLYNOMIAL DERIVATIVE FEATURES ───
    for window in [14, 28, 56]:
        poly_feats = grouped['close'].transform(
            lambda x: _compute_poly_features(x, window)
        )
        # This returns a single series; we need to compute separately
        pass  # Handled below with a dedicated function

    _add_poly_features(panel, grouped, windows=[14, 28, 56])

    # ─── 7. MOVING AVERAGE RATIOS ───
    for fast, slow in [(5, 20), (10, 50), (20, 100)]:
        sma_fast = grouped['close'].transform(lambda x: x.rolling(fast).mean())
        sma_slow = grouped['close'].transform(lambda x: x.rolling(slow).mean())
        panel[f'ma_ratio_{fast}_{slow}'] = sma_fast / (sma_slow + 1e-10) - 1

    # ─── 8. AMIHUD ILLIQUIDITY ───
    panel['amihud_28d'] = grouped.apply(
        lambda g: (g['log_ret'].abs() / (g['volume'] * g['close'] + 1)).rolling(28).mean()
    ).droplevel(0).sort_index() * 1e6

    # ─── 9. LIQUIDITY FEATURES ───
    panel['spread_proxy'] = 2 * (panel['high'] - panel['low']) / (panel['high'] + panel['low'] + 1e-10)
    panel['spread_28d'] = grouped['spread_proxy'].transform(lambda x: x.rolling(28).mean())

    # ─── 10. FORWARD RETURN (target) ───
    for horizon in [7, 14, 28]:
        panel[f'fwd_ret_{horizon}d'] = grouped['close'].transform(
            lambda x: x.pct_change(horizon).shift(-horizon)
        )

    return panel


def _add_poly_features(panel, grouped, windows=[14, 28, 56]):
    """Add polynomial regression derivative features."""
    for window in windows:
        slope_list = []
        curve_list = []
        r2_list = []

        for sym, g in panel.groupby(level='symbol'):
            close = g['close'].values
            n = len(close)
            slopes = np.full(n, np.nan)
            curves = np.full(n, np.nan)
            r2s = np.full(n, np.nan)

            for i in range(window, n):
                y = np.log(close[i - window:i] + 1e-10)
                x = np.arange(window, dtype=float)
                x = (x - x.mean()) / (x.std() + 1e-10)  # normalize for stability

                try:
                    coeffs = np.polyfit(x, y, 2)
                    slopes[i] = coeffs[1]  # linear term
                    curves[i] = coeffs[0]  # quadratic term
                    y_pred = np.polyval(coeffs, x)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2s[i] = 1 - ss_res / (ss_tot + 1e-10)
                except Exception:
                    pass

            slope_list.append(pd.Series(slopes, index=g.index))
            curve_list.append(pd.Series(curves, index=g.index))
            r2_list.append(pd.Series(r2s, index=g.index))

        panel[f'poly_slope_{window}d'] = pd.concat(slope_list)
        panel[f'poly_curve_{window}d'] = pd.concat(curve_list)
        panel[f'poly_r2_{window}d'] = pd.concat(r2_list)

    # Derived: velocity at endpoint
    for window in windows:
        panel[f'poly_velocity_{window}d'] = (
            panel[f'poly_slope_{window}d'] + 2 * panel[f'poly_curve_{window}d']
        )

    # Derived: high return + negative curvature = pullback signal
    for window in [28, 56]:
        panel[f'pullback_signal_{window}d'] = (
            panel[f'ret_{window}d'].clip(lower=0) *
            (-panel[f'poly_curve_{window}d']).clip(lower=0) *
            panel[f'rvol_{min(window, 28)}d']
        )


def _compute_poly_features(series, window):
    """Helper for poly features (unused, replaced by _add_poly_features)."""
    return series  # placeholder


def cross_sectional_rank(panel, feature_cols):
    """Rank features cross-sectionally per date."""
    for col in feature_cols:
        panel[f'{col}_rank'] = panel.groupby(level='date')[col].rank(pct=True)
    return panel


def build_feature_matrix(resample='1D', min_candles=250):
    """Main entry point: load data → build panel → compute features."""
    print("Loading data...")
    data = load_all_data(min_candles)
    print(f"Loaded {len(data)} symbols")

    print(f"Building panel (resample={resample})...")
    panel = build_panel(data, resample=resample)
    print(f"Panel shape: {panel.shape}")

    print("Computing features...")
    panel = compute_features(panel)

    # Drop rows with too many NaNs
    feature_cols = [c for c in panel.columns if c not in
                    ['open', 'high', 'low', 'close', 'volume', 'quote_vol',
                     'log_ret', 'spread_proxy', 'atr_14d']]
    panel = panel.dropna(subset=['close'])

    print(f"Final panel: {panel.shape}, {len(panel.index.get_level_values('symbol').unique())} symbols")
    return panel, data


if __name__ == '__main__':
    panel, data = build_feature_matrix(resample='1D')
    # Save
    out_path = os.path.join(DATA_DIR, 'feature_panel.parquet')
    panel.to_parquet(out_path)
    print(f"Saved to {out_path}")
