"""
Feature Engine v2 — 150+ Features for Cross-Sectional Pairwise Ranking
=======================================================================
Massive feature expansion based on VertoxQuant LTR framework.
Regularization in the model handles the noise.

Categories:
  1. Momentum Variants (20+)
  2. Volatility Regime (15+)
  3. Volume/Liquidity Advanced (12+)
  4. Mean Reversion (12+)
  5. Technical Extended (15+)
  6. Polynomial on Smoothed MA (30+) — Committee 2
  7. MA Derivatives (20+) — Committee 2
  8. Risk/Tail (10+)
  9. Interaction/Composite (15+)
  10. Forward targets
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from numba import njit, prange

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
UNIVERSE_DIR = os.path.join(DATA_DIR, 'universe')


# ════════════════════════════════════════════
# NUMBA HELPERS
# ════════════════════════════════════════════

@njit(cache=True)
def _ema_1d(arr, span):
    """Exponential moving average for a 1D array."""
    n = len(arr)
    out = np.empty(n)
    alpha = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        if np.isnan(arr[i]):
            out[i] = out[i-1]
        else:
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]
    return out


@njit(cache=True)
def _sma_1d(arr, window):
    """Simple moving average for a 1D array."""
    n = len(arr)
    out = np.full(n, np.nan)
    cumsum = 0.0
    for i in range(n):
        cumsum += arr[i]
        if i >= window:
            cumsum -= arr[i - window]
        if i >= window - 1:
            out[i] = cumsum / window
    return out


@njit(cache=True)
def _rolling_std(arr, window):
    """Rolling standard deviation."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        segment = arr[i - window + 1:i + 1]
        m = 0.0
        for v in segment:
            m += v
        m /= window
        var = 0.0
        for v in segment:
            var += (v - m) ** 2
        out[i] = (var / window) ** 0.5
    return out


@njit(cache=True)
def _poly_fit_window(y, window):
    """Quadratic poly fit on last `window` points. Returns (slope, curve, r2, velocity)."""
    n = len(y)
    slopes = np.full(n, np.nan)
    curves = np.full(n, np.nan)
    r2s = np.full(n, np.nan)
    velocities = np.full(n, np.nan)

    # Precompute normalized x
    x = np.arange(window, dtype=np.float64)
    xm = 0.0
    for v in x: xm += v
    xm /= window
    xs = 0.0
    for v in x: xs += (v - xm) ** 2
    xs = (xs / window) ** 0.5
    if xs < 1e-10:
        return slopes, curves, r2s, velocities
    xn = (x - xm) / xs

    # Precompute (X'X)^-1 X' for [xn^2, xn, 1]
    # Manual 3x3 least squares
    xx = np.empty((window, 3))
    for i in range(window):
        xx[i, 0] = xn[i] * xn[i]
        xx[i, 1] = xn[i]
        xx[i, 2] = 1.0

    # X'X
    xtx = np.zeros((3, 3))
    for i in range(window):
        for r in range(3):
            for c in range(3):
                xtx[r, c] += xx[i, r] * xx[i, c]

    # Invert 3x3
    det = (xtx[0,0]*(xtx[1,1]*xtx[2,2]-xtx[1,2]*xtx[2,1])
          -xtx[0,1]*(xtx[1,0]*xtx[2,2]-xtx[1,2]*xtx[2,0])
          +xtx[0,2]*(xtx[1,0]*xtx[2,1]-xtx[1,1]*xtx[2,0]))
    if abs(det) < 1e-20:
        return slopes, curves, r2s, velocities

    inv = np.zeros((3, 3))
    inv[0,0] = (xtx[1,1]*xtx[2,2]-xtx[1,2]*xtx[2,1]) / det
    inv[0,1] = -(xtx[0,1]*xtx[2,2]-xtx[0,2]*xtx[2,1]) / det
    inv[0,2] = (xtx[0,1]*xtx[1,2]-xtx[0,2]*xtx[1,1]) / det
    inv[1,0] = -(xtx[1,0]*xtx[2,2]-xtx[1,2]*xtx[2,0]) / det
    inv[1,1] = (xtx[0,0]*xtx[2,2]-xtx[0,2]*xtx[2,0]) / det
    inv[1,2] = -(xtx[0,0]*xtx[1,2]-xtx[0,2]*xtx[1,0]) / det
    inv[2,0] = (xtx[1,0]*xtx[2,1]-xtx[1,1]*xtx[2,0]) / det
    inv[2,1] = -(xtx[0,0]*xtx[2,1]-xtx[0,1]*xtx[2,0]) / det
    inv[2,2] = (xtx[0,0]*xtx[1,1]-xtx[0,1]*xtx[1,0]) / det

    # Pseudo-inverse = inv @ X'
    pinv = np.zeros((3, window))
    for r in range(3):
        for j in range(window):
            for c in range(3):
                pinv[r, j] += inv[r, c] * xx[j, c]

    for i in range(window, n):
        seg = y[i-window:i]
        has_nan = False
        for v in seg:
            if np.isnan(v):
                has_nan = True
                break
        if has_nan:
            continue

        # beta = pinv @ seg
        beta = np.zeros(3)
        for r in range(3):
            for j in range(window):
                beta[r] += pinv[r, j] * seg[j]

        curves[i] = beta[0]
        slopes[i] = beta[1]

        # R²
        ym = 0.0
        for v in seg: ym += v
        ym /= window
        ss_tot = 0.0
        ss_res = 0.0
        for j in range(window):
            pred = beta[0]*xn[j]*xn[j] + beta[1]*xn[j] + beta[2]
            ss_res += (seg[j] - pred) ** 2
            ss_tot += (seg[j] - ym) ** 2
        r2s[i] = 1.0 - ss_res / (ss_tot + 1e-15) if ss_tot > 1e-15 else 0.0
        velocities[i] = beta[1] + 2.0 * beta[0]

    return slopes, curves, r2s, velocities


@njit(cache=True)
def _diff_1d(arr):
    """First difference."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(1, n):
        out[i] = arr[i] - arr[i-1]
    return out


@njit(cache=True)
def _rolling_max(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window-1, n):
        mx = arr[i-window+1]
        for j in range(i-window+2, i+1):
            if arr[j] > mx:
                mx = arr[j]
        out[i] = mx
    return out


@njit(cache=True)
def _rolling_min(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window-1, n):
        mn = arr[i-window+1]
        for j in range(i-window+2, i+1):
            if arr[j] < mn:
                mn = arr[j]
        out[i] = mn
    return out


# ════════════════════════════════════════════
# DATA LOADING (same as v1)
# ════════════════════════════════════════════

def load_all_data(min_candles=250):
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
    frames = []
    for sym, df in data_dict.items():
        daily = df.resample(resample).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna(subset=['close'])
        if 'quote_vol' in df.columns:
            daily['quote_vol'] = df['quote_vol'].resample(resample).sum()
        else:
            daily['quote_vol'] = daily['volume'] * daily['close']
        daily['symbol'] = sym
        frames.append(daily)

    panel = pd.concat(frames).reset_index().rename(columns={'open_time': 'date', 'index': 'date'})
    if 'date' not in panel.columns:
        panel = panel.reset_index().rename(columns={panel.index.name or 'index': 'date'})
    panel = panel.set_index(['date', 'symbol']).sort_index()
    return panel


# ════════════════════════════════════════════
# PER-SYMBOL FEATURE COMPUTATION (vectorized)
# ════════════════════════════════════════════

def compute_symbol_features(close, high, low, volume, quote_vol):
    """Compute ALL features for a single symbol. Returns dict of feature arrays."""
    n = len(close)
    feats = {}
    log_close = np.log(close + 1e-10)
    log_ret = np.diff(log_close, prepend=log_close[0])

    # ════════════════════════════════════════
    # 1. MOMENTUM VARIANTS (25 features)
    # ════════════════════════════════════════
    for lb in [3, 5, 7, 10, 14, 21, 28, 35, 42, 56, 90, 120]:
        ret = np.full(n, np.nan)
        for i in range(lb, n):
            ret[i] = close[i] / close[i - lb] - 1.0
        feats[f'ret_{lb}d'] = ret

    # Skip-1 momentum
    for lb in [14, 21, 28, 35, 56]:
        ret = np.full(n, np.nan)
        for i in range(lb + 1, n):
            ret[i] = close[i - 1] / close[i - 1 - lb] - 1.0
        feats[f'ret_{lb}d_skip1'] = ret

    # Risk-adjusted momentum (ret / vol)
    for lb in [14, 28, 56]:
        vol = _rolling_std(log_ret, lb)
        ret = feats[f'ret_{lb}d']
        ra = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(ret[i]) and not np.isnan(vol[i]) and vol[i] > 1e-10:
                ra[i] = ret[i] / vol[i]
        feats[f'ret_{lb}d_riskadjusted'] = ra

    # Momentum acceleration (short - long)
    for s, l in [(7, 28), (14, 56), (28, 90)]:
        if f'ret_{s}d' in feats and f'ret_{l}d' in feats:
            feats[f'mom_accel_{s}_{l}'] = feats[f'ret_{s}d'] - feats[f'ret_{l}d']

    # Momentum z-score (ret / rolling_std_of_ret)
    for lb in [14, 28]:
        ret = feats[f'ret_{lb}d']
        ret_std = _rolling_std(ret, 60)
        zs = np.full(n, np.nan)
        ret_mean = _sma_1d(ret, 60)
        for i in range(n):
            if not np.isnan(ret[i]) and not np.isnan(ret_std[i]) and ret_std[i] > 1e-10:
                zs[i] = (ret[i] - (ret_mean[i] if not np.isnan(ret_mean[i]) else 0.0)) / ret_std[i]
        feats[f'ret_{lb}d_zscore'] = zs

    # ════════════════════════════════════════
    # 2. VOLATILITY REGIME (18 features)
    # ════════════════════════════════════════
    for w in [7, 14, 28, 56, 90]:
        feats[f'rvol_{w}d'] = _rolling_std(log_ret, w) * np.sqrt(365)

    # Vol-of-vol
    rvol28 = feats['rvol_28d']
    feats['vol_of_vol_28'] = _rolling_std(rvol28, 28)

    # Vol term structure (short / long)
    for s, l in [(7, 28), (14, 56), (28, 90)]:
        sv = feats[f'rvol_{s}d']
        lv = feats[f'rvol_{l}d']
        feats[f'vol_ratio_{s}_{l}'] = np.where(np.abs(lv) > 1e-10, sv / lv, np.nan)

    # Vol z-score
    vol_mean = _sma_1d(rvol28, 90)
    vol_std = _rolling_std(rvol28, 90)
    feats['vol_zscore'] = np.where(vol_std > 1e-10, (rvol28 - vol_mean) / vol_std, 0.0)

    # Skew & kurtosis
    for w in [14, 28, 56]:
        skew = np.full(n, np.nan)
        kurt = np.full(n, np.nan)
        for i in range(w, n):
            seg = log_ret[i-w:i]
            m = np.mean(seg)
            s = np.std(seg)
            if s > 1e-10:
                skew[i] = np.mean(((seg - m) / s) ** 3)
                kurt[i] = np.mean(((seg - m) / s) ** 4) - 3.0
        feats[f'skew_{w}d'] = skew
        feats[f'kurt_{w}d'] = kurt

    # Max/min returns
    for w in [14, 28]:
        feats[f'max_ret_{w}d'] = _rolling_max(log_ret, w)
        feats[f'min_ret_{w}d'] = _rolling_min(log_ret, w)

    # Downside vol
    neg_ret = np.where(log_ret < 0, log_ret, 0.0)
    feats['downvol_28d'] = _rolling_std(neg_ret, 28) * np.sqrt(365)

    # ════════════════════════════════════════
    # 3. VOLUME / LIQUIDITY (15 features)
    # ════════════════════════════════════════
    log_vol = np.log(quote_vol + 1.0)

    for w in [7, 14, 28]:
        feats[f'vol_avg_{w}d'] = _sma_1d(quote_vol, w)

    feats['vol_ratio_7_28'] = np.where(
        feats['vol_avg_28d'] > 1, feats['vol_avg_7d'] / feats['vol_avg_28d'], np.nan)

    # Volume surprise (today vs 28d avg)
    feats['vol_surprise'] = np.where(
        feats['vol_avg_28d'] > 1, quote_vol / feats['vol_avg_28d'] - 1.0, 0.0)

    # Volume trend (slope of log volume)
    feats['vol_trend_14d'] = np.full(n, np.nan)
    feats['vol_trend_28d'] = np.full(n, np.nan)
    for w, name in [(14, 'vol_trend_14d'), (28, 'vol_trend_28d')]:
        for i in range(w, n):
            seg = log_vol[i-w:i]
            x = np.arange(w, dtype=np.float64)
            mx = np.mean(x)
            my = np.mean(seg)
            num = np.sum((x - mx) * (seg - my))
            den = np.sum((x - mx) ** 2)
            feats[name][i] = num / den if den > 1e-10 else 0.0

    # Amihud illiquidity
    amihud_raw = np.abs(log_ret) / (quote_vol + 1.0) * 1e6
    feats['amihud_14d'] = _sma_1d(amihud_raw, 14)
    feats['amihud_28d'] = _sma_1d(amihud_raw, 28)

    # Spread proxy
    spread = 2.0 * (high - low) / (high + low + 1e-10)
    feats['spread_14d'] = _sma_1d(spread, 14)
    feats['spread_28d'] = _sma_1d(spread, 28)

    # Turnover proxy
    turnover = volume / (close + 1e-10)
    feats['turnover_14d'] = _sma_1d(turnover, 14)
    feats['turnover_28d'] = _sma_1d(turnover, 28)

    # Volume-price correlation (14d rolling)
    feats['vol_price_corr_14'] = np.full(n, np.nan)
    for i in range(14, n):
        p_seg = log_ret[i-14:i]
        v_seg = log_vol[i-14:i]
        pm, vm = np.mean(p_seg), np.mean(v_seg)
        cov = np.mean((p_seg - pm) * (v_seg - vm))
        sp = np.std(p_seg) * np.std(v_seg)
        feats['vol_price_corr_14'][i] = cov / sp if sp > 1e-10 else 0.0

    # ════════════════════════════════════════
    # 4. MEAN REVERSION (12 features)
    # ════════════════════════════════════════
    for w in [10, 20, 50, 100]:
        ma = _sma_1d(close, w)
        feats[f'dist_sma_{w}'] = (close - ma) / (ma + 1e-10)

    for w in [10, 20, 50]:
        ema = _ema_1d(close, w)
        feats[f'dist_ema_{w}'] = (close - ema) / (ema + 1e-10)

    # Z-score vs rolling mean
    for w in [20, 50]:
        ma = _sma_1d(close, w)
        std = _rolling_std(close, w)
        feats[f'zscore_{w}d'] = np.where(std > 1e-10, (close - ma) / std, 0.0)

    # Distance from high/low
    for w in [20, 50]:
        rmax = _rolling_max(close, w)
        rmin = _rolling_min(close, w)
        rng = rmax - rmin
        feats[f'range_pos_{w}d'] = np.where(rng > 1e-10, (close - rmin) / rng, 0.5)

    # ════════════════════════════════════════
    # 5. TECHNICALS EXTENDED (20 features)
    # ════════════════════════════════════════
    # RSI at multiple periods
    for p in [7, 14, 28]:
        delta = _diff_1d(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = _ema_1d(gain, p)
        avg_loss = _ema_1d(loss, p)
        rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
        feats[f'rsi_{p}d'] = 100.0 - 100.0 / (1.0 + rs)

    # Bollinger %B and width
    for w in [20, 50]:
        ma = _sma_1d(close, w)
        std = _rolling_std(close, w)
        feats[f'bb_pctb_{w}'] = np.where(std > 1e-10, (close - (ma - 2*std)) / (4*std + 1e-10), 0.5)
        feats[f'bb_width_{w}'] = np.where(ma > 1e-10, 4 * std / ma, 0.0)

    # MACD
    ema12 = _ema_1d(close, 12)
    ema26 = _ema_1d(close, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema_1d(macd_line, 9)
    feats['macd_hist'] = (macd_line - macd_signal) / (close + 1e-10)
    feats['macd_ratio'] = macd_line / (close + 1e-10)

    # Stochastic %K at multiple periods
    for p in [14, 28]:
        hi = _rolling_max(high, p)
        lo = _rolling_min(low, p)
        feats[f'stoch_{p}d'] = np.where((hi - lo) > 1e-10, (close - lo) / (hi - lo), 0.5)

    # CCI
    tp = (high + low + close) / 3.0
    tp_sma = _sma_1d(tp, 20)
    tp_mad = np.full(n, np.nan)
    for i in range(20, n):
        seg = tp[i-20:i]
        tp_mad[i] = np.mean(np.abs(seg - np.mean(seg)))
    feats['cci_20'] = np.where(tp_mad > 1e-10, (tp - tp_sma) / (0.015 * tp_mad + 1e-10), 0.0)

    # ATR ratio
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    tr[0] = high[0] - low[0]
    feats['atr_ratio_14'] = _sma_1d(tr, 14) / (close + 1e-10)

    # MA ratios
    for fast, slow in [(5, 20), (10, 50), (20, 100), (50, 200)]:
        if n > slow:
            sma_f = _sma_1d(close, fast)
            sma_s = _sma_1d(close, slow)
            feats[f'ma_ratio_{fast}_{slow}'] = np.where(sma_s > 1e-10, sma_f / sma_s - 1.0, 0.0)

    # ADX proxy
    rvol14 = feats.get('rvol_14d', np.zeros(n))
    ret14 = feats.get('ret_14d', np.zeros(n))
    feats['adx_proxy'] = np.clip(np.abs(ret14) * rvol14, 0, 5)

    # ════════════════════════════════════════
    # 6. POLYNOMIAL ON SMOOTHED MAs (30 features)
    # ════════════════════════════════════════

    # Poly on raw log(close)
    for w in [14, 28, 56, 90]:
        s, c, r, v = _poly_fit_window(log_close, w)
        feats[f'poly_slope_{w}d'] = s
        feats[f'poly_curve_{w}d'] = c
        feats[f'poly_r2_{w}d'] = r
        feats[f'poly_velocity_{w}d'] = v

    # Poly on EMA-smoothed close (Committee 2: key insight)
    for ema_span in [20, 50]:
        smoothed = np.log(_ema_1d(close, ema_span) + 1e-10)
        for w in [14, 28, 56]:
            s, c, r, v = _poly_fit_window(smoothed, w)
            feats[f'poly_ema{ema_span}_slope_{w}d'] = s
            feats[f'poly_ema{ema_span}_curve_{w}d'] = c
            feats[f'poly_ema{ema_span}_r2_{w}d'] = r

    # Poly on SMA-smoothed close
    for sma_w in [20, 50]:
        smoothed = np.log(_sma_1d(close, sma_w) + 1e-10)
        for w in [28, 56]:
            s, c, r, v = _poly_fit_window(smoothed, w)
            feats[f'poly_sma{sma_w}_slope_{w}d'] = s
            feats[f'poly_sma{sma_w}_curve_{w}d'] = c

    # ════════════════════════════════════════
    # 7. MA DERIVATIVES (20 features)
    # ════════════════════════════════════════

    # First derivative of MA (slope)
    for w in [10, 20, 50, 100]:
        ma = _sma_1d(close, w)
        # d/dt MA ≈ (MA_t - MA_{t-1}) / MA_t (normalized)
        ma_slope = _diff_1d(ma) / (ma + 1e-10)
        feats[f'ma{w}_slope'] = ma_slope

        # Second derivative (acceleration)
        ma_accel = _diff_1d(ma_slope)
        feats[f'ma{w}_accel'] = ma_accel

    # EMA derivatives
    for span in [10, 20, 50]:
        ema = _ema_1d(close, span)
        ema_slope = _diff_1d(ema) / (ema + 1e-10)
        feats[f'ema{span}_slope'] = ema_slope
        feats[f'ema{span}_accel'] = _diff_1d(ema_slope)

    # Cross-MA derivative divergence: short MA slope vs long MA slope
    for s, l in [(10, 50), (20, 100)]:
        if f'ma{s}_slope' in feats and f'ma{l}_slope' in feats:
            feats[f'ma_slope_div_{s}_{l}'] = feats[f'ma{s}_slope'] - feats[f'ma{l}_slope']

    # Poly on volume (volume trend)
    for w in [28, 56]:
        s, c, r, v = _poly_fit_window(log_vol, w)
        feats[f'vol_poly_slope_{w}d'] = s
        feats[f'vol_poly_curve_{w}d'] = c

    # Poly on volatility (vol regime trend)
    rvol_series = feats.get('rvol_28d', np.zeros(n))
    log_rvol = np.log(np.maximum(rvol_series, 1e-10))
    for w in [28, 56]:
        s, c, r, v = _poly_fit_window(log_rvol, w)
        feats[f'volreg_poly_slope_{w}d'] = s
        feats[f'volreg_poly_curve_{w}d'] = c

    # ════════════════════════════════════════
    # 8. RISK / TAIL (12 features)
    # ════════════════════════════════════════

    # Drawdown from peak
    for w in [28, 56, 90]:
        peak = _rolling_max(close, w)
        feats[f'drawdown_{w}d'] = (close - peak) / (peak + 1e-10)

    # Drawdown speed (how fast it's falling)
    dd28 = feats.get('drawdown_28d', np.zeros(n))
    feats['dd_speed_7d'] = np.full(n, np.nan)
    for i in range(7, n):
        feats['dd_speed_7d'][i] = dd28[i] - dd28[i-7]

    # Max favorable/adverse excursion
    for w in [14, 28]:
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        for i in range(w, n):
            entry = close[i - w]
            if entry > 1e-10:
                seg = close[i-w:i]
                mfe[i] = (np.max(seg) / entry - 1.0)
                mae[i] = (np.min(seg) / entry - 1.0)
        feats[f'mfe_{w}d'] = mfe
        feats[f'mae_{w}d'] = mae

    # Tail ratio (positive tail / negative tail)
    feats['tail_ratio_28'] = np.full(n, np.nan)
    for i in range(28, n):
        seg = log_ret[i-28:i]
        pos = seg[seg > 0]
        neg = seg[seg < 0]
        if len(neg) > 0 and len(pos) > 0:
            feats['tail_ratio_28'][i] = np.mean(pos) / (-np.mean(neg) + 1e-10)

    # ════════════════════════════════════════
    # 9. INTERACTIONS / COMPOSITE (15 features)
    # ════════════════════════════════════════

    # Momentum × Volume (confirmed trends)
    for lb in [14, 28]:
        ret = feats.get(f'ret_{lb}d', np.zeros(n))
        vr = feats.get('vol_ratio_7_28', np.ones(n))
        feats[f'mom_x_vol_{lb}d'] = ret * np.where(np.isnan(vr), 1.0, vr)

    # Momentum × Low volatility (quality momentum)
    for lb in [14, 28]:
        ret = feats.get(f'ret_{lb}d', np.zeros(n))
        vol = feats.get(f'rvol_{lb}d', np.ones(n))
        feats[f'quality_mom_{lb}d'] = np.where(vol > 1e-10, ret / vol, 0.0)

    # Poly slope × R² (trend confidence)
    for w in [28, 56]:
        slope = feats.get(f'poly_slope_{w}d', np.zeros(n))
        r2 = feats.get(f'poly_r2_{w}d', np.zeros(n))
        feats[f'trend_conf_{w}d'] = slope * np.where(np.isnan(r2), 0.0, r2)

    # Poly curve × Vol (volatile acceleration = danger)
    for w in [28, 56]:
        curve = feats.get(f'poly_curve_{w}d', np.zeros(n))
        vol = feats.get(f'rvol_{min(w, 28)}d', np.ones(n))
        feats[f'curve_x_vol_{w}d'] = curve * vol

    # Pullback signal (high return + negative curvature + high vol)
    for w in [28, 56]:
        ret = feats.get(f'ret_{w}d', np.zeros(n))
        curve = feats.get(f'poly_curve_{w}d', np.zeros(n))
        vol = feats.get('rvol_28d', np.ones(n))
        feats[f'pullback_{w}d'] = np.clip(ret, 0, None) * np.clip(-curve, 0, None) * vol

    # Mean reversion × Vol regime
    dist20 = feats.get('dist_sma_20', np.zeros(n))
    vol_z = feats.get('vol_zscore', np.zeros(n))
    feats['mr_x_volregime'] = dist20 * vol_z

    # ════════════════════════════════════════
    # 10. COMMITTEE ADDITIONS — Tier 1 features
    # ════════════════════════════════════════

    # --- MA 2nd derivative (closed-form, Committee 2 Tier 1) ---
    for w in [20, 50, 100]:
        d2 = np.full(n, np.nan)
        for i in range(w + 1, n):
            d2[i] = (close[i] - close[i-1] - close[i-w] + close[i-w-1]) / (w * close[i] + 1e-10)
        feats[f'd2_sma{w}'] = d2

    # --- MA 3rd derivative (jerk — inflection detector) ---
    for w in [20, 50]:
        d2 = feats.get(f'd2_sma{w}', np.zeros(n))
        feats[f'd3_sma{w}'] = _diff_1d(d2)

    # --- Cross-timeframe poly slope divergence ---
    for s, l in [(14, 56), (28, 90)]:
        ss = feats.get(f'poly_slope_{s}d', np.zeros(n))
        sl = feats.get(f'poly_slope_{l}d', np.zeros(n))
        feats[f'poly_slope_div_{s}_{l}'] = ss - sl

    # --- Poly on relative strength vs BTC (if BTC exists) ---
    # Will be computed at panel level later (needs cross-symbol data)

    # --- Cubic polynomial (S-curve detection) ---
    for w in [28, 56]:
        cubic = np.full(n, np.nan)
        for i in range(w, n):
            seg = log_close[i-w:i]
            if np.any(np.isnan(seg)):
                continue
            x = np.arange(w, dtype=np.float64)
            xm, xs_val = np.mean(x), np.std(x)
            if xs_val < 1e-10:
                continue
            xn = (x - xm) / xs_val
            try:
                coeffs = np.polyfit(xn, seg, 3)
                cubic[i] = coeffs[0]  # cubic term
            except Exception:
                pass
        feats[f'poly3_cubic_{w}d'] = cubic

    # --- Exponentially-weighted poly (WLS) ---
    for w in [28, 56]:
        ew_slope = np.full(n, np.nan)
        ew_curve = np.full(n, np.nan)
        halflife = w // 2
        lam = np.log(2.0) / halflife
        x = np.arange(w, dtype=np.float64)
        xm, xs_val = np.mean(x), np.std(x)
        if xs_val > 1e-10:
            xn = (x - xm) / xs_val
            ew = np.exp(lam * (x - w + 1))  # recent = higher weight
            sqw = np.sqrt(ew)
            X = np.column_stack([xn**2, xn, np.ones(w)])
            for i in range(w, n):
                seg = log_close[i-w:i]
                if np.any(np.isnan(seg)):
                    continue
                Xw = X * sqw[:, None]
                yw = seg * sqw
                try:
                    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
                    ew_curve[i] = beta[0]
                    ew_slope[i] = beta[1]
                except Exception:
                    pass
        feats[f'poly_ewt_slope_{w}d'] = ew_slope
        feats[f'poly_ewt_curve_{w}d'] = ew_curve

    # --- Poly residual (deviation from own trend) ---
    for w in [28, 56]:
        resid_last = np.full(n, np.nan)
        resid_std = np.full(n, np.nan)
        x = np.arange(w, dtype=np.float64)
        xm, xs_val = np.mean(x), np.std(x)
        if xs_val > 1e-10:
            xn = (x - xm) / xs_val
            X = np.column_stack([xn**2, xn, np.ones(w)])
            pinv = np.linalg.pinv(X)
            for i in range(w, n):
                seg = log_close[i-w:i]
                if np.any(np.isnan(seg)):
                    continue
                beta = pinv @ seg
                pred = X @ beta
                resid = seg - pred
                rs = np.std(resid)
                resid_std[i] = rs / (np.std(seg) + 1e-10)
                resid_last[i] = resid[-1] / (rs + 1e-10)
        feats[f'poly_resid_std_{w}d'] = resid_std
        feats[f'poly_resid_last_{w}d'] = resid_last

    # --- Microstructure (Committee 1) ---
    feats['close_location_value'] = (close - low) / (high - low + 1e-10)
    feats['clv_sma_7d'] = _sma_1d(feats['close_location_value'], 7)

    upper_shadow = (high - np.maximum(close, np.roll(close, 1))) / (high - low + 1e-10)
    upper_shadow[0] = 0
    feats['upper_shadow_7d'] = _sma_1d(np.clip(upper_shadow, 0, 1), 7)

    lower_shadow = (np.minimum(close, np.roll(close, 1)) - low) / (high - low + 1e-10)
    lower_shadow[0] = 0
    feats['lower_shadow_7d'] = _sma_1d(np.clip(lower_shadow, 0, 1), 7)

    body_ratio = np.abs(close - np.roll(close, 1)) / (high - low + 1e-10)
    body_ratio[0] = 0
    feats['body_ratio_7d'] = _sma_1d(body_ratio, 7)

    # Efficiency ratio (Kaufman)
    for w in [14, 28]:
        net_move = np.full(n, np.nan)
        total_path = np.full(n, np.nan)
        for i in range(w, n):
            net_move[i] = abs(close[i] - close[i - w])
            total_path[i] = np.sum(np.abs(np.diff(close[i-w:i+1])))
        feats[f'efficiency_{w}d'] = np.where(total_path > 1e-10, net_move / total_path, 0.0)

    # --- Risk/Tail advanced (Committee 1) ---
    # MFE minus return (gave-back)
    for w in [14, 28]:
        mfe = feats.get(f'mfe_{w}d', np.zeros(n))
        ret = feats.get(f'ret_{w}d', np.zeros(n))
        feats[f'mfe_minus_ret_{w}d'] = mfe - ret

    # Gain-pain ratio
    feats['gain_pain_28d'] = np.full(n, np.nan)
    for i in range(28, n):
        seg = log_ret[i-28:i]
        gains = np.sum(seg[seg > 0])
        pains = abs(np.sum(seg[seg < 0]))
        feats['gain_pain_28d'][i] = gains / (pains + 1e-10)

    # Ulcer index
    feats['ulcer_28d'] = np.full(n, np.nan)
    for i in range(28, n):
        seg = close[i-28:i+1]
        peak = np.maximum.accumulate(seg)
        dd = (seg - peak) / (peak + 1e-10)
        feats['ulcer_28d'][i] = np.sqrt(np.mean(dd ** 2))

    # --- EWM features (Committee 1, time-decay) ---
    for hl in [5, 14, 28]:
        alpha = 1 - np.exp(-np.log(2) / hl)
        ewm_ret = np.full(n, np.nan)
        ewm_ret[0] = log_ret[0]
        for i in range(1, n):
            ewm_ret[i] = alpha * log_ret[i] + (1 - alpha) * ewm_ret[i-1]
        feats[f'ewm_ret_hl{hl}'] = ewm_ret

    # EWM vol
    for hl in [7, 21]:
        alpha = 1 - np.exp(-np.log(2) / hl)
        ewm_var = np.full(n, np.nan)
        ewm_var[0] = log_ret[0] ** 2
        for i in range(1, n):
            ewm_var[i] = alpha * log_ret[i]**2 + (1 - alpha) * ewm_var[i-1]
        feats[f'ewm_vol_hl{hl}'] = np.sqrt(np.maximum(ewm_var, 0)) * np.sqrt(365)

    # EWM vol ratio
    if 'ewm_vol_hl7' in feats and 'ewm_vol_hl21' in feats:
        feats['ewm_vol_ratio_7_21'] = feats['ewm_vol_hl7'] / (feats['ewm_vol_hl21'] + 1e-10)

    # --- Nonlinear transforms (Committee 1) ---
    feats['log_amihud_28d'] = np.log(np.maximum(feats.get('amihud_28d', np.ones(n)), 1e-10))
    feats['log_turnover_28d'] = np.log(np.maximum(feats.get('turnover_28d', np.ones(n)), 1e-10))
    for lb in [14, 28]:
        ret = feats.get(f'ret_{lb}d', np.zeros(n))
        feats[f'sigmoid_ret_{lb}d'] = 2.0 / (1.0 + np.exp(-ret * 10)) - 1.0

    # Momentum consistency
    for w in [28, 56]:
        cons = np.full(n, np.nan)
        for i in range(w, n):
            seg = log_ret[i-w:i]
            cons[i] = np.sum(seg > 0) / w
        feats[f'mom_consistency_{w}d'] = cons

    # Split-window poly slope comparison
    for w in [28, 56]:
        half = w // 2
        split_diff = np.full(n, np.nan)
        for i in range(w, n):
            seg1 = log_close[i-w:i-half]
            seg2 = log_close[i-half:i]
            if np.any(np.isnan(seg1)) or np.any(np.isnan(seg2)):
                continue
            x1 = np.arange(half, dtype=np.float64)
            x2 = np.arange(half, dtype=np.float64)
            try:
                s1 = np.polyfit(x1, seg1, 1)[0]
                s2 = np.polyfit(x2, seg2, 1)[0]
                split_diff[i] = s2 - s1
            except Exception:
                pass
        feats[f'poly_split_slope_diff_{w}d'] = split_diff

    # ════════════════════════════════════════
    # 11. FORWARD RETURNS (targets)
    # ════════════════════════════════════════
    for h in [7, 14, 28]:
        fwd = np.full(n, np.nan)
        for i in range(n - h):
            fwd[i] = close[i + h] / close[i] - 1.0
        feats[f'fwd_ret_{h}d'] = fwd

    return feats


# ════════════════════════════════════════════
# MAIN BUILD
# ════════════════════════════════════════════

def build_feature_matrix_v2(resample='1D', min_candles=250):
    """Load data → build panel → compute 150+ features."""
    print("Loading data...")
    data = load_all_data(min_candles)
    print(f"  {len(data)} symbols")

    print("Building panel...")
    panel = build_panel(data, resample)
    print(f"  Panel: {panel.shape}")

    print("Computing 150+ features per symbol...")
    all_feats = {}
    symbols = panel.index.get_level_values('symbol').unique()
    n_syms = len(symbols)

    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n_syms}] {sym}")

        try:
            grp = panel.xs(sym, level='symbol')
            close = grp['close'].values.astype(np.float64)
            high = grp['high'].values.astype(np.float64)
            low = grp['low'].values.astype(np.float64)
            volume = grp['volume'].values.astype(np.float64)
            qvol = grp.get('quote_vol', grp['volume'] * grp['close']).values.astype(np.float64)

            feats = compute_symbol_features(close, high, low, volume, qvol)

            for fname, fvals in feats.items():
                series = pd.Series(fvals, index=grp.index, name=fname)
                if fname not in all_feats:
                    all_feats[fname] = []
                all_feats[fname].append(
                    pd.DataFrame({'date': grp.index, 'symbol': sym, fname: fvals}).set_index(['date', 'symbol'])
                )
        except Exception as e:
            print(f"    WARN: {sym} failed: {e}")
            continue

    # Merge all features into panel
    print("Merging features into panel...")
    for fname, frames in all_feats.items():
        merged = pd.concat(frames)
        panel = panel.join(merged, how='left')

    # Clean up
    panel = panel.dropna(subset=['close'])

    feat_cols = [c for c in panel.columns if c not in
                 ['open', 'high', 'low', 'close', 'volume', 'quote_vol',
                  'log_ret', 'spread_proxy', 'atr_14d', 'symbol']
                 and not c.startswith('fwd_ret_')]
    feat_cols = [c for c in feat_cols if panel[c].dtype in ('float64', 'float32')]

    print(f"\nFinal: {panel.shape}, {len(feat_cols)} features")
    print(f"Feature categories:")
    cats = {}
    for f in feat_cols:
        cat = f.split('_')[0] if '_' in f else f
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cat}: {count}")

    return panel, feat_cols


if __name__ == '__main__':
    panel, feat_cols = build_feature_matrix_v2()
    out_path = os.path.join(DATA_DIR, 'feature_panel_v2.parquet')
    panel.to_parquet(out_path)
    print(f"\nSaved to {out_path}")
    print(f"Features: {len(feat_cols)}")
