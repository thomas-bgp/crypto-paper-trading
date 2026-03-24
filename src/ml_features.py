"""
ML Feature Engine — computes 60+ factors for the cross-sectional model.
Optimized: computes per-symbol in batch, then stacks into panel.
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
UNIVERSE_DIR = os.path.join(DATA_DIR, 'universe')


def load_daily_panel(min_candles=200):
    """Load all coins, resample to daily, return stacked panel."""
    frames = []
    for base_dir in [UNIVERSE_DIR, DATA_DIR]:
        if not os.path.exists(base_dir):
            continue
        for f in os.listdir(base_dir):
            if not f.endswith('_4h.parquet') or f.startswith('_'):
                continue
            sym = f.replace('_4h.parquet', '')
            # Skip duplicates
            seen_syms = {fr['symbol'].iloc[0] for fr in frames if 'symbol' in fr.columns and len(fr) > 0}
            if sym in seen_syms:
                continue
            try:
                df = pd.read_parquet(os.path.join(base_dir, f))
                if len(df) < min_candles:
                    continue
                # Resample to daily
                daily = df.resample('1D').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum',
                }).dropna(subset=['close'])
                if 'quote_vol' in df.columns:
                    daily['quote_vol'] = df['quote_vol'].resample('1D').sum()
                else:
                    daily['quote_vol'] = daily['volume'] * daily['close']
                daily['symbol'] = sym
                frames.append(daily)
            except Exception:
                continue

    panel = pd.concat(frames).reset_index().rename(columns={'open_time': 'date', 'index': 'date'})
    if 'date' not in panel.columns:
        panel = panel.reset_index().rename(columns={panel.index.name or 'index': 'date'})
    panel = panel.set_index(['date', 'symbol']).sort_index()
    print(f"Panel: {panel.shape[0]} rows, {panel.index.get_level_values('symbol').nunique()} symbols")
    return panel


def compute_all_features(panel):
    """Compute ~60 features vectorized per symbol group."""
    g = panel.groupby(level='symbol')
    close = panel['close']
    high = panel['high']
    low = panel['low']
    volume = panel['volume']
    qvol = panel['quote_vol']

    # ── Log return ──
    panel['log_ret'] = g['close'].transform(lambda x: np.log(x / x.shift(1)))

    # ════════════════════════════════════════════
    # 1. MOMENTUM (12 features)
    # ════════════════════════════════════════════
    for lb in [7, 14, 21, 28, 56, 90]:
        panel[f'mom_{lb}'] = g['close'].pct_change(lb)
    # Skip-1 (reversal avoidance)
    for lb in [14, 28]:
        panel[f'mom_{lb}_skip1'] = g['close'].shift(1).transform(lambda x: x.pct_change(lb))
    # Robust momentum: mean of 21-35d
    mom_cols = [f'mom_{lb}' for lb in [21, 28]]
    panel['mom_robust'] = panel[mom_cols].mean(axis=1)
    # 52-week high ratio
    panel['high_52w'] = close / g['close'].transform(lambda x: x.rolling(252).max())
    # Momentum acceleration
    panel['mom_accel'] = panel['mom_14'] - panel['mom_28']

    # ════════════════════════════════════════════
    # 2. VOLATILITY & HIGHER MOMENTS (12 features)
    # ════════════════════════════════════════════
    lr = panel['log_ret']
    for w in [14, 28, 56]:
        panel[f'rvol_{w}'] = g['log_ret'].transform(lambda x: x.rolling(w).std() * np.sqrt(365))
    panel['vol_of_vol'] = g['rvol_28'].transform(lambda x: x.rolling(28).std())
    panel['skew_28'] = g['log_ret'].transform(lambda x: x.rolling(28).skew())
    panel['kurt_28'] = g['log_ret'].transform(lambda x: x.rolling(28).kurt())
    panel['max_ret_28'] = g['log_ret'].transform(lambda x: x.rolling(28).max())
    panel['min_ret_28'] = g['log_ret'].transform(lambda x: x.rolling(28).min())
    # Downside vol
    neg_ret = panel['log_ret'].clip(upper=0)
    panel['downvol_28'] = neg_ret.groupby(level='symbol').transform(lambda x: x.rolling(28).std() * np.sqrt(365))
    # Vol ratio (short/long)
    panel['vol_ratio'] = panel['rvol_14'] / (panel['rvol_56'] + 1e-10)
    # Upside/downside vol ratio
    panel['up_down_vol'] = panel['rvol_28'] / (panel['downvol_28'] + 1e-10)

    # ════════════════════════════════════════════
    # 3. VOLUME & LIQUIDITY (8 features)
    # ════════════════════════════════════════════
    panel['vol_avg_28'] = g['quote_vol'].transform(lambda x: x.rolling(28).mean())
    panel['vol_ratio_7_28'] = g['quote_vol'].transform(lambda x: x.rolling(7).mean()) / (panel['vol_avg_28'] + 1)
    panel['turnover'] = qvol / (close * 1e6 + 1)  # proxy
    panel['turnover_28'] = g['turnover'].transform(lambda x: x.rolling(28).mean())
    # Amihud illiquidity
    panel['amihud'] = (panel['log_ret'].abs() / (qvol + 1)).groupby(level='symbol').transform(
        lambda x: x.rolling(28).mean()) * 1e9
    # Spread proxy
    panel['spread'] = 2 * (high - low) / (high + low + 1e-10)
    panel['spread_28'] = g['spread'].transform(lambda x: x.rolling(28).mean())
    # Volume trend
    panel['vol_mom'] = g['quote_vol'].pct_change(14)

    # ════════════════════════════════════════════
    # 4. TECHNICAL INDICATORS (14 features)
    # ════════════════════════════════════════════
    # RSI
    delta = g['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for p in [14]:
        ag = gain.groupby(level='symbol').transform(lambda x: x.ewm(span=p, adjust=False).mean())
        al = loss.groupby(level='symbol').transform(lambda x: x.ewm(span=p, adjust=False).mean())
        panel[f'rsi_{p}'] = 100 - 100 / (1 + ag / (al + 1e-10))

    # Bollinger %B & width
    sma20 = g['close'].transform(lambda x: x.rolling(20).mean())
    std20 = g['close'].transform(lambda x: x.rolling(20).std())
    panel['bb_pctb'] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-10)
    panel['bb_width'] = 4 * std20 / (sma20 + 1e-10)

    # MACD
    ema12 = g['close'].transform(lambda x: x.ewm(span=12).mean())
    ema26 = g['close'].transform(lambda x: x.ewm(span=26).mean())
    macd = ema12 - ema26
    signal = macd.groupby(level='symbol').transform(lambda x: x.ewm(span=9).mean())
    panel['macd_hist'] = (macd - signal) / (close + 1e-10)

    # Stochastic %K
    low14 = g['low'].transform(lambda x: x.rolling(14).min())
    high14 = g['high'].transform(lambda x: x.rolling(14).max())
    panel['stoch_k'] = (close - low14) / (high14 - low14 + 1e-10)

    # CCI
    tp = (high + low + close) / 3
    sma_tp = tp.groupby(level='symbol').transform(lambda x: x.rolling(20).mean())
    mad_tp = tp.groupby(level='symbol').transform(
        lambda x: x.rolling(20).apply(lambda v: np.abs(v - v.mean()).mean(), raw=True))
    panel['cci'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-10)

    # Donchian position
    h20 = g['high'].transform(lambda x: x.rolling(20).max())
    l20 = g['low'].transform(lambda x: x.rolling(20).min())
    panel['donchian_pos'] = (close - l20) / (h20 - l20 + 1e-10)

    # ATR ratio
    tr = pd.concat([high - low, (high - g['close'].shift(1)).abs(),
                    (low - g['close'].shift(1)).abs()], axis=1).max(axis=1)
    panel['atr_ratio'] = tr.groupby(level='symbol').transform(
        lambda x: x.rolling(14).mean()) / (close + 1e-10)

    # MA ratios
    for fast, slow in [(10, 50), (20, 100)]:
        sf = g['close'].transform(lambda x: x.rolling(fast).mean())
        ss = g['close'].transform(lambda x: x.rolling(slow).mean())
        panel[f'ma_{fast}_{slow}'] = sf / (ss + 1e-10) - 1

    # ADX proxy (simplified: abs of directional difference)
    panel['adx_proxy'] = (panel['mom_14'].abs() * panel['rvol_14']).clip(upper=5)

    # ════════════════════════════════════════════
    # 5. POLYNOMIAL DERIVATIVES (12 features)
    # ════════════════════════════════════════════
    for window in [14, 28, 56]:
        _compute_poly_batch(panel, window)

    # ════════════════════════════════════════════
    # 6. CROSS-SECTIONAL FEATURES (computed per date)
    # ════════════════════════════════════════════
    # These require groupby date
    panel['mom_14_csrank'] = panel.groupby(level='date')['mom_14'].rank(pct=True)
    panel['mom_28_csrank'] = panel.groupby(level='date')['mom_28'].rank(pct=True)
    panel['rvol_28_csrank'] = panel.groupby(level='date')['rvol_28'].rank(pct=True)
    panel['vol_avg_28_csrank'] = panel.groupby(level='date')['vol_avg_28'].rank(pct=True)

    # ════════════════════════════════════════════
    # 7. TARGETS (forward returns)
    # ════════════════════════════════════════════
    for h in [7, 14, 28]:
        panel[f'fwd_{h}'] = g['close'].transform(lambda x: x.pct_change(h).shift(-h))
    # Ranked target (cross-sectional percentile of fwd return)
    for h in [14]:
        panel[f'fwd_{h}_rank'] = panel.groupby(level='date')[f'fwd_{h}'].rank(pct=True)

    return panel


def _compute_poly_batch(panel, window):
    """Compute polynomial slope, curvature, velocity for all symbols."""
    results = {f'poly_slope_{window}': [], f'poly_curve_{window}': [],
               f'poly_r2_{window}': [], f'poly_velocity_{window}': []}

    for sym, grp in panel.groupby(level='symbol'):
        c = np.log(grp['close'].values + 1e-10)
        n = len(c)
        slope = np.full(n, np.nan)
        curve = np.full(n, np.nan)
        r2 = np.full(n, np.nan)

        # Vectorized sliding window via stride tricks would be ideal
        # but polyfit needs loop — keep it tight
        x = np.arange(window, dtype=np.float64)
        x_norm = (x - x.mean()) / (x.std() + 1e-10)
        X = np.column_stack([x_norm**2, x_norm, np.ones(window)])  # [t², t, 1]

        # Precompute pseudo-inverse for speed (same X every window)
        XtX_inv_Xt = np.linalg.pinv(X)

        for i in range(window, n):
            y = c[i-window:i]
            if np.any(np.isnan(y)):
                continue
            # β = (X'X)⁻¹ X'y — precomputed pseudo-inverse
            beta = XtX_inv_Xt @ y
            slope[i] = beta[1]
            curve[i] = beta[0]
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2[i] = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0

        idx = grp.index
        results[f'poly_slope_{window}'].append(pd.Series(slope, index=idx))
        results[f'poly_curve_{window}'].append(pd.Series(curve, index=idx))
        results[f'poly_r2_{window}'].append(pd.Series(r2, index=idx))
        velocity = slope + 2 * curve  # instantaneous velocity at endpoint
        results[f'poly_velocity_{window}'].append(pd.Series(velocity, index=idx))

    for col, series_list in results.items():
        panel[col] = pd.concat(series_list)

    # Derived: pullback signal (high return + negative curvature + high vol)
    panel[f'pullback_{window}'] = (
        panel[f'mom_{min(window, 28)}'].clip(lower=0) *
        (-panel[f'poly_curve_{window}']).clip(lower=0) *
        panel['rvol_28']
    )


def get_feature_columns(panel):
    """Return list of feature columns (excluding targets, OHLCV, etc)."""
    exclude = {'open', 'high', 'low', 'close', 'volume', 'quote_vol',
               'log_ret', 'spread', 'turnover', 'symbol',
               'vol_avg_28', 'atr_ratio'}
    exclude.update(c for c in panel.columns if c.startswith('fwd_'))
    return [c for c in panel.columns if c not in exclude and panel[c].dtype in ('float64', 'float32')]


def build_ml_dataset():
    """Main entry: load → compute → return panel + feature names."""
    print("Loading daily panel...")
    panel = load_daily_panel()
    print("Computing features...")
    panel = compute_all_features(panel)
    feat_cols = get_feature_columns(panel)
    print(f"Features: {len(feat_cols)}")
    print(f"Sample features: {feat_cols[:10]}")
    return panel, feat_cols


if __name__ == '__main__':
    panel, feat_cols = build_ml_dataset()
    print(f"\nFinal: {panel.shape}, {len(feat_cols)} features")
    print(f"Features: {feat_cols}")
    # Save
    out = os.path.join(DATA_DIR, 'ml_panel.parquet')
    panel.to_parquet(out)
    print(f"Saved to {out}")
