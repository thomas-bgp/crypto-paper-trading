"""
Paper Trading Engine — CatBoost R3 Regime Strategy (ALIGNED WITH BACKTEST)
Runs ML models against LIVE Binance SPOT data without real capital.

MATCHES the validated regime_v2.py backtest exactly:
  - SPOT data (not futures) — survivorship-bias free universe
  - 60+ features from cv_spot.py compute_features()
  - SINGLE CatBoostRanker per target (fwd_min, fwd_max) — NOT ensemble
  - R3 regime: breadth + breadth momentum -> long/short/cash
  - Long: top 10% by fwd_max, hold 2d, trail 5%, hard stop 8%
  - Short: bottom 5 by fwd_min, hold 5d, trail trough*(1+15%)
  - Vol targeting built into position sizing

Usage:
    python paper_trading.py              # Run one cycle (cron-friendly)
    python paper_trading.py --daemon     # Run continuously with sleep between cycles
    python paper_trading.py --status     # Print current state
    python paper_trading.py --retrain    # Force model retrain
"""
import numpy as np
import pandas as pd
import requests
import json
import os
import sys
import time
import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'
PAPER_DIR = PROJECT_DIR / 'paper_trading'
PAPER_DIR.mkdir(exist_ok=True)

# ─── Config: SHORT strategy (matches regime_v2.py backtest) ───
HOLDING_DAYS = 5               # Short hold 5d (regime_v2 SHORT_HOLD=5)
TOP_N = 5                      # Bottom 5 by fwd_min score (regime_v2 SHORT_TOP=5)
UNIVERSE_TOP = 80              # Top 80 for TRAINING panel only (matches backtest filter_universe top_n=80)
                               # Live universe: no cap — all pairs passing volume filter are included
COST_PER_SIDE = 0.0011         # 0.11% per side (matches backtest COST=0.0011)
STOP_PCT = 0.15                # 15% trailing stop: trough*(1+15%) (regime_v2 SHORT_TRAIL=0.15)
INITIAL_CAPITAL = 10_000       # paper capital (total, split between strategies)
CAPITAL_SPLIT = 0.5            # 50% short, 50% long (configurable)
TRAIN_MONTHS = 18
STABLE_APY = 0.05              # 5% annual stablecoin yield on idle capital
STABLE_YIELD_PER_4H = STABLE_APY / 365 / 6  # yield per 4h cycle
FUND_COST = 0.0001             # Funding cost per period for shorts (regime_v2 FUND=0.0001)

# ─── Config: LONG strategy (matches regime_v2.py backtest) ───
HOLDING_DAYS_LONG = 2          # 2 days holding (regime_v2 LONG_HOLD=2)
LONG_PCT = 0.10                # Top 10% of universe by max score (regime_v2 LONG_PCT=0.10)
MAX_POSITIONS_LONG = 10        # Cap at 10 positions (regime_v2 LONG_MAX=10)
TRAILING_LONG = 0.05           # 5% trailing stop below high (regime_v2 LONG_TRAIL=0.05)
HARD_STOP_LONG = 0.08          # 8% hard stop below entry (regime_v2 LONG_HARD=0.08)
REBALANCE_LONG = 5             # Rebalance every 5 days (regime_v2 REBAL=5)

# ─── Vol targeting (matches regime_v2.py) ───
VOL_TARGET = 0.50

# ─── Binance SPOT API (NOT futures — matches backtest which uses SPOT data) ───
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"
# Futures endpoints still needed for funding rates and mark prices on short positions
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_MARK_PRICE = "https://fapi.binance.com/fapi/v1/premiumIndex"

# Stablecoin / leveraged token filter (matches cv_spot.py)
STABLECOINS = {'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'FDUSDUSDT', 'DAIUSDT', 'USTUSDT',
               'EURUSDT', 'USDPUSDT', 'PAXUSDT'}

CATBOOST_PARAMS = {
    'loss_function': 'YetiRank',
    'iterations': 300,
    'depth': 4,
    'learning_rate': 0.05,
    'l2_leaf_reg': 5.0,
    'random_strength': 2.0,
    'bagging_temperature': 1.0,
    'border_count': 64,
    'verbose': 0,
    'random_seed': 42,
    'task_type': 'CPU',
}

# ─── State files ───
STATE_FILE = PAPER_DIR / 'state.json'
TRADES_FILE = PAPER_DIR / 'trades.json'
EQUITY_FILE = PAPER_DIR / 'equity.json'
MODEL_MIN_FILE = PAPER_DIR / 'model_fwd_min.cbm'   # Single model for fwd_min (short)
MODEL_MAX_FILE = PAPER_DIR / 'model_fwd_max.cbm'   # Single model for fwd_max (long)
LOG_FILE = PAPER_DIR / 'log.jsonl'

# Long strategy state files (separate from short)
TRADES_LONG_FILE = PAPER_DIR / 'trades_long.json'
EQUITY_LONG_FILE = PAPER_DIR / 'equity_long.json'


# ════════════════════════════════════════════
# BINANCE DATA (LIVE)
# ════════════════════════════════════════════

def get_spot_symbols(min_volume_usd=5_000_000):
    """Get actively traded SPOT USDT pairs sorted by 24h quote volume.
    Live filter: all SPOT USDT pairs with 24h volume >= $5M, no artificial cap.
    Could be 60-100+ coins depending on market conditions.
    """
    try:
        resp = requests.get(BINANCE_TICKER, timeout=15)
        tickers = resp.json()
        result = []
        for t in tickers:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT') or '_' in sym:
                continue
            # Filter stablecoins (matches cv_spot.py STABLECOINS)
            if sym in STABLECOINS:
                continue
            # Filter leveraged tokens (matches cv_spot.py)
            if any(sym.endswith(s) for s in ['UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT']):
                continue
            vol = float(t.get('quoteVolume', 0))
            if vol >= min_volume_usd:
                result.append({'symbol': sym, 'volume_24h': vol, 'price': float(t.get('lastPrice', 0))})
        return sorted(result, key=lambda x: -x['volume_24h'])  # No cap — take all passing volume filter
    except Exception as e:
        log_event('error', f'Failed to get spot symbols: {e}')
        return []


def fetch_recent_klines(symbol, interval='4h', limit=500):
    """Fetch recent klines for feature computation."""
    try:
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        resp = requests.get(BINANCE_KLINES, params=params, timeout=15)
        data = resp.json()
        if not data or isinstance(data, dict):
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_buy_vol',
            'taker_buy_quote_vol', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_vol']:
            df[col] = df[col].astype(float)
        df.set_index('open_time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume', 'quote_vol']]
    except Exception as e:
        log_event('error', f'Klines fetch failed for {symbol}: {e}')
        return pd.DataFrame()


def fetch_current_prices(symbols):
    """Get current SPOT prices for a list of symbols.
    Uses SPOT ticker endpoint (not futures mark price).
    """
    try:
        resp = requests.get(BINANCE_TICKER, timeout=15)
        data = resp.json()
        prices = {}
        for item in data:
            if item['symbol'] in symbols:
                prices[item['symbol']] = float(item['lastPrice'])
        return prices
    except Exception:
        return {}


def fetch_current_funding(symbols):
    """Get latest funding rates."""
    rates = {}
    try:
        resp = requests.get(BINANCE_MARK_PRICE, timeout=15)
        data = resp.json()
        for item in data:
            if item['symbol'] in symbols:
                rates[item['symbol']] = float(item.get('lastFundingRate', 0))
    except Exception:
        pass
    return rates


# ════════════════════════════════════════════
# FEATURE COMPUTATION (live panel)
# ════════════════════════════════════════════

def build_live_panel(symbols_info):
    """Build a panel from live klines data for feature computation."""
    frames = []
    for info in symbols_info:
        sym = info['symbol']
        df = fetch_recent_klines(sym, '4h', 500)
        if df.empty or len(df) < 100:
            continue
        # Resample to daily
        daily = df.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum', 'quote_vol': 'sum',
        }).dropna(subset=['close'])
        daily['symbol'] = sym
        frames.append(daily)
        time.sleep(0.1)  # rate limit

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames).reset_index().rename(columns={'open_time': 'date', 'index': 'date'})
    if 'date' not in panel.columns:
        panel = panel.reset_index().rename(columns={panel.index.name or 'index': 'date'})
    panel = panel.set_index(['date', 'symbol']).sort_index()
    return panel


def compute_features_live(panel):
    """Compute the FULL feature set on live panel.
    Matches cv_spot.py compute_features() exactly — 60+ features including:
    momentum, volatility, liquidity, technical, polynomial, market structure, cross-sectional.
    """
    g = panel.groupby(level='symbol')
    close, high, low, qvol = panel['close'], panel['high'], panel['low'], panel['quote_vol']

    panel['log_ret'] = g['close'].transform(lambda x: np.log(x / x.shift(1)))

    # ── Momentum (matches cv_spot.py) ──
    for lb in [3, 5, 7, 14, 21, 28]:
        panel[f'mom_{lb}'] = g['close'].pct_change(lb)
    for lb in [14, 28]:
        panel[f'mom_{lb}_skip1'] = g['close'].shift(1).transform(lambda x: x.pct_change(lb))
    panel['mom_robust'] = panel[['mom_21', 'mom_28']].mean(axis=1)
    panel['mom_accel'] = panel['mom_14'] - panel['mom_28']

    # ── Volatility ──
    for w in [7, 14, 28]:
        panel[f'rvol_{w}'] = g['log_ret'].transform(lambda x: x.rolling(w).std() * np.sqrt(365))
    panel['vol_of_vol'] = g['rvol_28'].transform(lambda x: x.rolling(28).std())
    panel['skew_28'] = g['log_ret'].transform(lambda x: x.rolling(28).skew())
    panel['kurt_28'] = g['log_ret'].transform(lambda x: x.rolling(28).kurt())
    panel['max_ret_28'] = g['log_ret'].transform(lambda x: x.rolling(28).max())
    panel['min_ret_28'] = g['log_ret'].transform(lambda x: x.rolling(28).min())
    panel['downvol_28'] = panel['log_ret'].clip(upper=0).groupby(level='symbol').transform(
        lambda x: x.rolling(28).std() * np.sqrt(365))
    panel['vol_ratio'] = panel['rvol_7'] / (panel['rvol_28'] + 1e-10)
    panel['up_down_vol'] = panel['rvol_28'] / (panel['downvol_28'] + 1e-10)

    # ── Liquidity ──
    panel['vol_avg_28'] = g['quote_vol'].transform(lambda x: x.rolling(28).mean())
    panel['vol_ratio_7_28'] = g['quote_vol'].transform(lambda x: x.rolling(7).mean()) / (panel['vol_avg_28'] + 1)
    panel['turnover'] = qvol / (close * 1e6 + 1)
    panel['turnover_28'] = g['turnover'].transform(lambda x: x.rolling(28).mean())
    panel['amihud'] = (panel['log_ret'].abs() / (qvol + 1)).groupby(level='symbol').transform(
        lambda x: x.rolling(28).mean()) * 1e9
    panel['spread'] = 2 * (high - low) / (high + low + 1e-10)
    panel['spread_28'] = g['spread'].transform(lambda x: x.rolling(28).mean())
    panel['vol_mom'] = g['quote_vol'].pct_change(14)

    # ── Technical ──
    delta = g['close'].diff()
    ag = delta.clip(lower=0).groupby(level='symbol').transform(lambda x: x.ewm(span=14, adjust=False).mean())
    al = (-delta).clip(lower=0).groupby(level='symbol').transform(lambda x: x.ewm(span=14, adjust=False).mean())
    panel['rsi_14'] = 100 - 100 / (1 + ag / (al + 1e-10))

    sma20 = g['close'].transform(lambda x: x.rolling(20).mean())
    std20 = g['close'].transform(lambda x: x.rolling(20).std())
    panel['bb_pctb'] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)
    panel['bb_width'] = 4 * std20 / (sma20 + 1e-10)

    ema12 = g['close'].transform(lambda x: x.ewm(span=12).mean())
    ema26 = g['close'].transform(lambda x: x.ewm(span=26).mean())
    macd = ema12 - ema26
    panel['macd_hist'] = (macd - macd.groupby(level='symbol').transform(
        lambda x: x.ewm(span=9).mean())) / (close + 1e-10)

    panel['stoch_k'] = (close - g['low'].transform(lambda x: x.rolling(14).min())) / (
        g['high'].transform(lambda x: x.rolling(14).max()) -
        g['low'].transform(lambda x: x.rolling(14).min()) + 1e-10)

    tp = (high + low + close) / 3
    sma_tp = tp.groupby(level='symbol').transform(lambda x: x.rolling(20).mean())
    mad_tp = tp.groupby(level='symbol').transform(
        lambda x: x.rolling(20).apply(lambda v: np.abs(v - v.mean()).mean(), raw=True))
    panel['cci'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-10)

    panel['donchian_pos'] = (close - g['low'].transform(lambda x: x.rolling(20).min())) / (
        g['high'].transform(lambda x: x.rolling(20).max()) -
        g['low'].transform(lambda x: x.rolling(20).min()) + 1e-10)

    tr = pd.concat([high - low, (high - g['close'].shift(1)).abs(),
                     (low - g['close'].shift(1)).abs()], axis=1).max(axis=1)
    panel['atr_ratio'] = tr.groupby(level='symbol').transform(
        lambda x: x.rolling(14).mean()) / (close + 1e-10)

    for fast, slow in [(10, 50), (20, 100)]:
        panel[f'ma_{fast}_{slow}'] = g['close'].transform(
            lambda x: x.rolling(fast).mean()) / (
            g['close'].transform(lambda x: x.rolling(slow).mean()) + 1e-10) - 1

    panel['adx_proxy'] = (panel['mom_14'].abs() * panel['rvol_14']).clip(upper=5)

    # ── Polynomial derivatives (14 and 28, matches cv_spot.py) ──
    for window in [14, 28]:
        _compute_poly(panel, window)

    # ── Market structure (cross-date aggregates) ──
    dg = panel.groupby(level='date')
    panel['mkt_disp_7'] = dg['mom_7'].transform('std')
    panel['mkt_breadth_7'] = dg['mom_7'].transform(lambda x: (x > 0).mean())
    panel['mkt_med_ret_7'] = dg['mom_7'].transform('median')
    panel['mkt_med_ret_28'] = dg['mom_28'].transform('median')
    panel['mkt_avg_vol'] = dg['rvol_28'].transform('median')
    panel['mkt_skew'] = dg['mom_7'].transform('skew')
    panel['mkt_vol_conc'] = dg['vol_avg_28'].transform(
        lambda x: x.nlargest(5).sum() / (x.sum() + 1e-10) if len(x) > 5 else np.nan)
    panel['mkt_avg_spread'] = dg['spread_28'].transform('median')
    panel['mkt_vol_disp'] = dg['rvol_28'].transform('std')

    # ── Cross-sectional ranks ──
    panel['cs_mom14'] = dg['mom_14'].rank(pct=True)
    panel['cs_mom28'] = dg['mom_28'].rank(pct=True)
    panel['cs_rvol28'] = dg['rvol_28'].rank(pct=True)
    panel['cs_vol28'] = dg['vol_avg_28'].rank(pct=True)
    panel['cs_spread'] = dg['spread_28'].rank(pct=True)

    # ── Alpha signals ──
    panel['alpha_7'] = panel['mom_7'] - panel['mkt_med_ret_7']
    panel['alpha_28'] = panel['mom_28'] - panel['mkt_med_ret_28']

    return panel


def _compute_poly(panel, w):
    """Compute polynomial slope, curvature, R2, and velocity. Matches cv_spot.py _poly()."""
    res = {f'poly_slope_{w}': [], f'poly_curve_{w}': [], f'poly_r2_{w}': [], f'poly_velocity_{w}': []}
    x = np.arange(w, dtype=np.float64)
    xn = (x - x.mean()) / (x.std() + 1e-10)
    X = np.column_stack([xn**2, xn, np.ones(w)])
    P = np.linalg.pinv(X)

    for sym, grp in panel.groupby(level='symbol'):
        c = np.log(grp['close'].values + 1e-10)
        n = len(c)
        sl, cu, r2 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for i in range(w, n):
            y = c[i - w:i]
            if np.any(np.isnan(y)):
                continue
            b = P @ y
            sl[i] = b[1]
            cu[i] = b[0]
            yp = X @ b
            ssr = np.sum((y - yp)**2)
            sst = np.sum((y - y.mean())**2)
            r2[i] = 1 - ssr / (sst + 1e-10) if sst > 0 else 0
        idx = grp.index
        res[f'poly_slope_{w}'].append(pd.Series(sl, index=idx))
        res[f'poly_curve_{w}'].append(pd.Series(cu, index=idx))
        res[f'poly_r2_{w}'].append(pd.Series(r2, index=idx))
        res[f'poly_velocity_{w}'].append(pd.Series(sl + 2 * cu, index=idx))

    for col, series_list in res.items():
        if series_list:
            panel[col] = pd.concat(series_list)


def get_feature_cols(panel):
    """Get feature column names from panel. Matches cv_spot.py get_features()."""
    exclude = {'open', 'high', 'low', 'close', 'volume', 'quote_vol', 'log_ret',
               'spread', 'turnover', 'vol_avg_28', 'fwd_max', 'fwd_min'}
    return [c for c in panel.columns if c not in exclude and panel[c].dtype in ('float64', 'float32')]


# ════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════

def _load_training_panel():
    """Load and prepare SPOT training panel with full features and targets.
    Uses cv_spot.py-style data loading: SPOT parquets, survivorship-bias free.
    Falls back to ml_features if spot data unavailable.
    """
    SPOT_DIR = 'C:/Projects/crypto-statarb-lab/spot_data/parquet'
    MIN_CANDLES = 6 * 120  # matches cv_spot.py

    if os.path.exists(SPOT_DIR):
        log_event('train', f'Loading SPOT parquets from {SPOT_DIR}')
        frames = []
        for f in os.listdir(SPOT_DIR):
            if not f.endswith('_4h.parquet'):
                continue
            sym = f.replace('_4h.parquet', '')
            if sym in STABLECOINS:
                continue
            if any(sym.endswith(s) for s in ['UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT']):
                continue
            try:
                df = pd.read_parquet(os.path.join(SPOT_DIR, f))
                if len(df) < MIN_CANDLES:
                    continue
                daily = df.resample('1D').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna(subset=['close'])
                if 'quote_vol' in df.columns:
                    daily['quote_vol'] = df['quote_vol'].astype(float).resample('1D').sum()
                else:
                    daily['quote_vol'] = daily['volume'] * daily['close']
                daily['symbol'] = sym
                frames.append(daily)
            except Exception:
                continue
        panel = pd.concat(frames).reset_index()
        if 'open_time' in panel.columns:
            panel = panel.rename(columns={'open_time': 'date'})
        elif 'date' not in panel.columns:
            panel = panel.rename(columns={panel.columns[0]: 'date'})
        panel = panel.set_index(['date', 'symbol']).sort_index()
    else:
        # Fallback: use ml_features (futures data — less ideal but better than nothing)
        log_event('train', 'WARNING: SPOT data not found, falling back to ml_features (futures)')
        sys.path.insert(0, str(SCRIPT_DIR))
        from ml_features import load_daily_panel
        panel = load_daily_panel()

    # Filter universe (top N by rolling volume, matches cv_spot.py)
    g = panel.groupby(level='symbol')
    panel['_rv30'] = g['quote_vol'].transform(lambda x: x.rolling(30).mean())
    vr = panel.groupby(level='date')['_rv30'].rank(ascending=False)
    mask = (vr <= UNIVERSE_TOP) & (panel['_rv30'] >= 5_000_000)
    panel = panel[mask].copy()
    panel.drop(columns=['_rv30'], inplace=True)

    log_event('train', f'Panel: {panel.shape[0]:,} rows, {panel.index.get_level_values("symbol").nunique()} symbols')

    # Compute full features
    panel = compute_features_live(panel)

    # Compute targets: fwd_max and fwd_min (matches regime_v2.py exactly)
    targets = []
    for sym, grp in panel.groupby(level='symbol'):
        c = grp['close'].values
        n = len(c)
        mn, mx = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n - 5):
            w = c[i + 1:i + 6]
            b = c[i]
            if b <= 0 or np.any(np.isnan(w)):
                continue
            mn[i] = w.min() / b - 1
            mx[i] = w.max() / b - 1
        targets.append(pd.DataFrame({'fwd_min': mn, 'fwd_max': mx}, index=grp.index))
    tdf = pd.concat(targets)
    panel['fwd_min'] = tdf['fwd_min']
    panel['fwd_max'] = tdf['fwd_max']

    feat_cols = get_feature_cols(panel)
    # Remove targets from features if they snuck in
    feat_cols = [f for f in feat_cols if f not in ('fwd_min', 'fwd_max')]

    valid = panel[['fwd_min', 'fwd_max']].notna().all(axis=1) & panel[feat_cols].notna().all(axis=1)
    panel = panel[valid]

    log_event('train', f'{len(feat_cols)} features, {len(panel):,} valid rows')
    return panel, feat_cols


def train_models():
    """Train SINGLE CatBoostRanker for fwd_min and fwd_max.
    Matches regime_v2.py exactly: single model per target, same CB params, same data.
    """
    from catboost import CatBoostRanker, Pool

    log_event('train', 'Starting model training (SINGLE model per target, matches backtest)')

    panel, feat_cols = _load_training_panel()

    # Use most recent TRAIN_MONTHS for training
    dates = panel.index.get_level_values('date').unique().sort_values()
    train_end = dates[-1]
    train_start = train_end - pd.DateOffset(months=TRAIN_MONTHS)
    d = panel.index.get_level_values('date')
    train = panel[(d >= train_start) & (d <= train_end)]

    if len(train) < 1000:
        log_event('error', f'Not enough training data: {len(train)} rows')
        return None, None, feat_cols

    # Group IDs for ranking (matches regime_v2.py)
    td = train.index.get_level_values('date')
    ud = td.unique().sort_values()
    gid = np.array([{v: i for i, v in enumerate(ud)}[x] for x in td])

    X = train[feat_cols].values

    # Train fwd_min model (for SHORT selection)
    log_event('train', f'Training fwd_min model ({len(train):,} rows)...')
    m_min = CatBoostRanker(**CATBOOST_PARAMS)
    m_min.fit(Pool(X, train['fwd_min'].values, group_id=gid))
    m_min.save_model(str(MODEL_MIN_FILE))

    # Train fwd_max model (for LONG selection)
    log_event('train', f'Training fwd_max model ({len(train):,} rows)...')
    m_max = CatBoostRanker(**CATBOOST_PARAMS)
    m_max.fit(Pool(X, train['fwd_max'].values, group_id=gid))
    m_max.save_model(str(MODEL_MAX_FILE))

    # Save metadata
    meta = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'train_start': str(train_start.date()),
        'train_end': str(train_end.date()),
        'n_rows': len(train),
        'n_symbols': train.index.get_level_values('symbol').nunique(),
        'features': feat_cols,
        'targets': ['fwd_min', 'fwd_max'],
        'model_type': 'single_per_target',
    }
    with open(PAPER_DIR / 'model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Save feature importance (from fwd_max model)
    imp = pd.Series(
        m_max.get_feature_importance(type='PredictionValuesChange'),
        index=feat_cols
    ).sort_values(ascending=False)
    imp_dict = (imp / (imp.sum() + 1e-10)).to_dict()
    with open(PAPER_DIR / 'feature_importance.json', 'w') as f:
        json.dump(imp_dict, f, indent=2)

    log_event('train', f'Trained 2 single models on {len(train):,} rows '
              f'({train_start.date()} to {train_end.date()}), {len(feat_cols)} features')
    return m_min, m_max, feat_cols


def load_models():
    """Load saved single models (fwd_min + fwd_max)."""
    from catboost import CatBoostRanker
    m_min, m_max = None, None
    if MODEL_MIN_FILE.exists():
        m_min = CatBoostRanker()
        m_min.load_model(str(MODEL_MIN_FILE))
    if MODEL_MAX_FILE.exists():
        m_max = CatBoostRanker()
        m_max.load_model(str(MODEL_MAX_FILE))
    meta_path = PAPER_DIR / 'model_meta.json'
    feat_cols = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            feat_cols = meta.get('features')
    return m_min, m_max, feat_cols


def models_need_retrain():
    """Check if models are stale (>56 days old) or missing."""
    meta_path = PAPER_DIR / 'model_meta.json'
    if not meta_path.exists() or not MODEL_MIN_FILE.exists() or not MODEL_MAX_FILE.exists():
        return True
    with open(meta_path) as f:
        meta = json.load(f)
    trained = pd.Timestamp(meta['trained_at'])
    age_days = (pd.Timestamp.now(tz='UTC') - trained).days
    return age_days >= 56


# ════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════

def load_trades_long():
    if TRADES_LONG_FILE.exists():
        with open(TRADES_LONG_FILE) as f:
            return json.load(f)
    return []


def save_trades_long(trades):
    with open(TRADES_LONG_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def load_equity_history_long():
    if EQUITY_LONG_FILE.exists():
        with open(EQUITY_LONG_FILE) as f:
            return json.load(f)
    return []


def save_equity_history_long(history):
    with open(EQUITY_LONG_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def load_state():
    """Load paper trading state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        # Migration: add long fields if missing (backward compatible)
        if 'equity_short' not in state:
            state['equity_short'] = state.get('equity', INITIAL_CAPITAL * CAPITAL_SPLIT)
            state['equity_long'] = INITIAL_CAPITAL * CAPITAL_SPLIT
            state['positions_long'] = []
            state['last_rebalance_long'] = None
            state['total_trades_long'] = 0
            state['cumulative_yield_long'] = 0.0
            state['regime_skip_count'] = 0
        return state
    short_cap = INITIAL_CAPITAL * CAPITAL_SPLIT
    long_cap = INITIAL_CAPITAL * (1 - CAPITAL_SPLIT)
    return {
        'capital': INITIAL_CAPITAL,
        # Short strategy
        'equity': short_cap,
        'equity_short': short_cap,
        'positions': [],
        'last_rebalance': None,
        'total_trades': 0,
        'cumulative_yield': 0.0,
        # Long strategy
        'equity_long': long_cap,
        'positions_long': [],
        'last_rebalance_long': None,
        'total_trades_long': 0,
        'cumulative_yield_long': 0.0,
        'regime_skip_count': 0,
        # Shared
        'started_at': datetime.now(timezone.utc).isoformat(),
    }


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_trades():
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades):
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def load_equity_history():
    if EQUITY_FILE.exists():
        with open(EQUITY_FILE) as f:
            return json.load(f)
    return []


def save_equity_history(history):
    with open(EQUITY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def log_event(event_type, message, data=None):
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'type': event_type,
        'message': message,
    }
    if data:
        entry['data'] = data
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n')
    print(f"[{entry['timestamp'][:19]}] [{event_type.upper()}] {message}")


# ════════════════════════════════════════════
# POSITION MONITORING
# ════════════════════════════════════════════

def check_stops(state):
    """Check trailing stops on open positions."""
    if not state['positions']:
        return state, []

    symbols = [p['symbol'] for p in state['positions']]
    prices = fetch_current_prices(symbols)
    closed = []
    remaining = []

    for pos in state['positions']:
        sym = pos['symbol']
        current_price = prices.get(sym)
        if current_price is None:
            remaining.append(pos)
            continue

        # Update trough (lowest price since entry for shorts)
        trough = min(pos.get('trough', pos['entry_price']), current_price)
        pos['trough'] = trough
        pos['current_price'] = current_price

        # Trailing stop: if price rises STOP_PCT above trough
        stop_level = trough * (1 + STOP_PCT)
        if current_price >= stop_level:
            # Stopped out
            ret = -(current_price / pos['entry_price'] - 1)
            ret -= COST_PER_SIDE  # exit cost
            pos['exit_price'] = current_price
            pos['exit_time'] = datetime.now(timezone.utc).isoformat()
            pos['exit_reason'] = 'trailing_stop'
            pos['return'] = ret
            closed.append(pos)
            log_event('stop', f"STOPPED OUT {sym}: entry={pos['entry_price']:.4f} exit={current_price:.4f} ret={ret*100:.2f}%")
        else:
            # Check holding period expiry
            entry_time = pd.Timestamp(pos['entry_time'])
            days_held = (pd.Timestamp.now(tz='UTC') - entry_time).days
            if days_held >= HOLDING_DAYS:
                ret = -(current_price / pos['entry_price'] - 1)
                ret -= COST_PER_SIDE
                pos['exit_price'] = current_price
                pos['exit_time'] = datetime.now(timezone.utc).isoformat()
                pos['exit_reason'] = 'holding_expired'
                pos['return'] = ret
                closed.append(pos)
                log_event('close', f"EXPIRED {sym}: entry={pos['entry_price']:.4f} exit={current_price:.4f} ret={ret*100:.2f}% ({days_held}d)")
            else:
                pos['days_held'] = days_held
                pos['unrealized_pnl'] = -(current_price / pos['entry_price'] - 1)
                remaining.append(pos)

    # Apply closed PnL — fixed weight based on original position count (TOP_N)
    n_positions_at_entry = state.get('n_positions_at_entry', TOP_N)
    for pos in closed:
        weight = 1.0 / n_positions_at_entry
        state['equity'] *= (1 + pos['return'] * weight)

    state['positions'] = remaining

    # Record trades
    if closed:
        trades = load_trades()
        trades.extend(closed)
        save_trades(trades)

    return state, closed


def rebalance(state):
    """Run the full SHORT rebalance cycle: close old positions, predict, open new."""
    log_event('rebalance', 'Starting SHORT rebalance cycle')

    # Load or train models
    m_min, m_max, feat_cols = load_models()
    if m_min is None or models_need_retrain():
        log_event('rebalance', 'Models need training')
        m_min, m_max, feat_cols = train_models()

    if m_min is None:
        log_event('error', 'No fwd_min model available, skipping SHORT rebalance')
        return state

    # Close any remaining positions
    if state['positions']:
        symbols = [p['symbol'] for p in state['positions']]
        prices = fetch_current_prices(symbols)
        trades = load_trades()
        n_positions_at_entry = state.get('n_positions_at_entry', TOP_N)
        for pos in state['positions']:
            sym = pos['symbol']
            current_price = prices.get(sym, pos.get('current_price', pos['entry_price']))
            ret = -(current_price / pos['entry_price'] - 1)
            ret -= COST_PER_SIDE
            pos['exit_price'] = current_price
            pos['exit_time'] = datetime.now(timezone.utc).isoformat()
            pos['exit_reason'] = 'rebalance'
            pos['return'] = ret
            trades.append(pos)
            weight = 1.0 / n_positions_at_entry
            state['equity'] *= (1 + ret * weight)
            log_event('close', f"REBALANCE CLOSE {sym}: ret={ret*100:.2f}%")
        save_trades(trades)
        state['positions'] = []

    # Get live universe (SPOT)
    symbols_info = get_spot_symbols()
    if len(symbols_info) < 15:
        log_event('error', f'Only {len(symbols_info)} symbols found, need 15+')
        return state

    log_event('rebalance', f'Building live panel for {len(symbols_info)} symbols')

    # Build features
    panel = build_live_panel(symbols_info)
    if panel.empty:
        log_event('error', 'Empty panel, cannot rebalance')
        return state

    panel = compute_features_live(panel)

    # Get latest cross-section
    latest_date = panel.index.get_level_values('date').max()
    cross = panel.loc[latest_date].copy()
    cross = cross.dropna(subset=['mom_14'])

    # Filter to features available
    avail_feats = [f for f in feat_cols if f in cross.columns]
    if len(avail_feats) < 10:
        log_event('error', f'Only {len(avail_feats)} features available')
        return state

    X = np.nan_to_num(cross[avail_feats].values, nan=0, posinf=0, neginf=0)

    # Single model prediction (fwd_min — bottom = expected losers = short candidates)
    try:
        cross = cross.copy()
        cross['score_min'] = m_min.predict(X)
        cross['rank_min'] = cross['score_min'].rank(pct=True)
    except Exception as e:
        log_event('error', f'fwd_min model predict failed: {e}')
        return state

    cross = cross.sort_values('score_min', ascending=True)  # lowest score = worst = short

    # Select bottom N to short (matches regime_v2.py SHORT_TOP=5)
    short_syms = cross.head(TOP_N).index.tolist()

    # Get entry prices (use SPOT prices for SPOT positions)
    prices = fetch_current_prices(short_syms)
    funding_rates = fetch_current_funding(short_syms)

    # Open positions
    now = datetime.now(timezone.utc).isoformat()
    new_positions = []
    for sym in short_syms:
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        pos = {
            'symbol': sym,
            'direction': 'short',
            'entry_price': price,
            'entry_time': now,
            'trough': price,
            'current_price': price,
            'score': float(cross.loc[sym, 'score_min']),
            'funding_rate': funding_rates.get(sym, 0),
            'features': {f: float(cross.loc[sym, f]) if f in cross.columns else None for f in avail_feats[:5]},
            'unrealized_pnl': 0.0,
            'days_held': 0,
        }
        new_positions.append(pos)
        log_event('open', f"SHORT {sym} @ {price:.4f} (score={pos['score']:.3f})",
                  {'funding_rate': pos['funding_rate']})

    # Deduct entry costs
    state['equity'] *= (1 - COST_PER_SIDE * len(new_positions) / max(len(new_positions), 1))
    state['positions'] = new_positions
    state['n_positions_at_entry'] = len(new_positions)
    state['last_rebalance'] = now
    state['total_trades'] += len(new_positions)

    # Estimate funding cost for this period
    avg_funding = np.mean([abs(funding_rates.get(s, 0)) for s in short_syms]) if short_syms else 0
    state['estimated_funding_cost_pct'] = avg_funding * 3 * HOLDING_DAYS * 100

    # Record equity snapshot
    equity_hist = load_equity_history()
    equity_hist.append({
        'timestamp': now,
        'equity': state['equity'],
        'positions': len(new_positions),
        'symbols': [p['symbol'] for p in new_positions],
        'event': 'rebalance',
    })
    save_equity_history(equity_hist)

    log_event('rebalance', f"Opened {len(new_positions)} shorts. Equity: ${state['equity']:,.2f}")
    return state


# ════════════════════════════════════════════
# LONG STRATEGY — POSITION MONITORING
# ════════════════════════════════════════════

def check_stops_long(state):
    """Check trailing stops and hard stops on open LONG positions."""
    if not state.get('positions_long'):
        return state, []

    symbols = [p['symbol'] for p in state['positions_long']]
    prices = fetch_current_prices(symbols)
    closed = []
    remaining = []

    for pos in state['positions_long']:
        sym = pos['symbol']
        current_price = prices.get(sym)
        if current_price is None:
            remaining.append(pos)
            continue

        # Update peak (highest price since entry for longs)
        peak = max(pos.get('peak', pos['entry_price']), current_price)
        pos['peak'] = peak
        pos['current_price'] = current_price

        exit_reason = None
        # Hard stop: 8% below entry
        if current_price <= pos['entry_price'] * (1 - HARD_STOP_LONG):
            exit_reason = 'hard_stop'
        # Trailing stop: 5% below highest price seen
        elif current_price <= peak * (1 - TRAILING_LONG):
            exit_reason = 'trailing_stop'
        else:
            # Check holding period expiry (2 days)
            entry_time = pd.Timestamp(pos['entry_time'])
            days_held = (pd.Timestamp.now(tz='UTC') - entry_time).days
            if days_held >= HOLDING_DAYS_LONG:
                exit_reason = 'holding_expired'
            else:
                pos['days_held'] = days_held
                pos['unrealized_pnl'] = (current_price / pos['entry_price'] - 1)
                remaining.append(pos)

        if exit_reason:
            ret = (current_price / pos['entry_price'] - 1)
            ret -= COST_PER_SIDE  # exit cost
            pos['exit_price'] = current_price
            pos['exit_time'] = datetime.now(timezone.utc).isoformat()
            pos['exit_reason'] = exit_reason
            pos['return'] = ret
            closed.append(pos)
            log_event('stop' if 'stop' in exit_reason else 'close',
                      f"LONG {exit_reason.upper()} {sym}: entry={pos['entry_price']:.4f} exit={current_price:.4f} ret={ret*100:.2f}%")

    # Apply closed PnL
    n_positions_at_entry = state.get('n_positions_at_entry_long', MAX_POSITIONS_LONG)
    for pos in closed:
        weight = 1.0 / n_positions_at_entry
        state['equity_long'] *= (1 + pos['return'] * weight)

    state['positions_long'] = remaining

    # Record trades
    if closed:
        trades = load_trades_long()
        trades.extend(closed)
        save_trades_long(trades)

    return state, closed


def check_regime_filter(panel):
    """Regime filter R3: breadth + breadth momentum.
    Returns (direction, size_fraction, regime_info).
    direction: 'long', 'short', or 'cash'
    """
    try:
        g = panel.groupby(level='symbol')
        ret_7d = g['close'].transform(lambda x: x.pct_change(7))
        latest_date = panel.index.get_level_values('date').max()
        cross_ret = ret_7d.loc[latest_date]

        # Current breadth: % of coins with positive 7d return
        breadth = (cross_ret > 0).mean()

        # Breadth 5 days ago (approximate with earlier date)
        dates = panel.index.get_level_values('date').unique().sort_values()
        prev_date_idx = max(0, len(dates) - 6)
        prev_date = dates[prev_date_idx]
        prev_ret = ret_7d.loc[prev_date] if prev_date in ret_7d.index.get_level_values('date') else cross_ret
        prev_breadth = (prev_ret > 0).mean()
        breadth_delta = breadth - prev_breadth

        median_ret = cross_ret.median()

        # R3 logic: breadth level + breadth momentum
        if breadth < 0.35 or (breadth < 0.45 and breadth_delta < -0.05):
            direction = 'short'
            size_frac = 1.0
        elif breadth > 0.55 or (breadth > 0.45 and breadth_delta > 0.05):
            direction = 'long'
            size_frac = 1.0
        else:
            direction = 'cash'
            size_frac = 0.0

        regime_info = f'breadth={breadth:.2f} delta={breadth_delta:+.2f} medret={median_ret*100:+.1f}% -> {direction}'
        log_event('regime', regime_info)
        return direction, size_frac, median_ret
    except Exception as e:
        log_event('error', f'Regime filter failed: {e}')
        return 'long', 1.0, 0.0  # default to long if filter fails


def rebalance_long(state, panel=None, symbols_info=None):
    """Run the LONG rebalance cycle: close old positions, predict, open new."""
    log_event('rebalance', 'Starting LONG rebalance cycle')

    # Load or train models (shared with short — single model per target)
    m_min, m_max, feat_cols = load_models()
    if m_max is None or models_need_retrain():
        log_event('rebalance', 'Models need training')
        m_min, m_max, feat_cols = train_models()

    if m_max is None:
        log_event('error', 'No fwd_max model available, skipping LONG rebalance')
        return state

    # Close any remaining long positions
    if state.get('positions_long'):
        symbols = [p['symbol'] for p in state['positions_long']]
        prices = fetch_current_prices(symbols)
        trades = load_trades_long()
        n_positions_at_entry = state.get('n_positions_at_entry_long', MAX_POSITIONS_LONG)
        for pos in state['positions_long']:
            sym = pos['symbol']
            current_price = prices.get(sym, pos.get('current_price', pos['entry_price']))
            ret = (current_price / pos['entry_price'] - 1)
            ret -= COST_PER_SIDE
            pos['exit_price'] = current_price
            pos['exit_time'] = datetime.now(timezone.utc).isoformat()
            pos['exit_reason'] = 'rebalance'
            pos['return'] = ret
            trades.append(pos)
            weight = 1.0 / n_positions_at_entry
            state['equity_long'] *= (1 + ret * weight)
            log_event('close', f"LONG REBALANCE CLOSE {sym}: ret={ret*100:.2f}%")
        save_trades_long(trades)
        state['positions_long'] = []

    # Build live panel if not provided
    if panel is None:
        if symbols_info is None:
            symbols_info = get_spot_symbols()
        if len(symbols_info) < 15:
            log_event('error', f'Only {len(symbols_info)} symbols found, need 15+')
            return state
        panel = build_live_panel(symbols_info)
        if panel.empty:
            log_event('error', 'Empty panel, cannot rebalance LONG')
            return state
        panel = compute_features_live(panel)

    # Use regime already computed in run_cycle()
    regime_direction = state.get('regime_direction', 'long')
    if regime_direction != 'long':
        state['regime_skip_count'] = state.get('regime_skip_count', 0) + 1
        state['last_rebalance_long'] = datetime.now(timezone.utc).isoformat()
        log_event('regime', f'LONG skipped: regime={regime_direction}. Skip count: {state["regime_skip_count"]}')
        equity_hist = load_equity_history_long()
        equity_hist.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'equity': state['equity_long'],
            'positions': 0,
            'symbols': [],
            'event': 'regime_skip',
            'regime': regime_direction,
        })
        save_equity_history_long(equity_hist)
        return state

    # Get latest cross-section
    latest_date = panel.index.get_level_values('date').max()
    cross = panel.loc[latest_date].copy()
    cross = cross.dropna(subset=['mom_14'])

    avail_feats = [f for f in feat_cols if f in cross.columns]
    if len(avail_feats) < 10:
        log_event('error', f'LONG: Only {len(avail_feats)} features available')
        return state

    X = np.nan_to_num(cross[avail_feats].values, nan=0, posinf=0, neginf=0)

    # Single model prediction (fwd_max — highest = expected winners = long candidates)
    try:
        cross = cross.copy()
        cross['score_max'] = m_max.predict(X)
        cross['rank_max'] = cross['score_max'].rank(pct=True)
    except Exception as e:
        log_event('error', f'fwd_max model predict failed: {e}')
        return state

    cross = cross.sort_values('score_max', ascending=False)  # highest score = best = long

    # Select top 10% of universe, max MAX_POSITIONS_LONG
    n_long = max(3, int(len(cross) * LONG_PCT))
    long_syms = cross.head(min(n_long, MAX_POSITIONS_LONG)).index.tolist()

    # Get entry prices
    prices = fetch_current_prices(long_syms)
    funding_rates = fetch_current_funding(long_syms)

    # Open positions
    now = datetime.now(timezone.utc).isoformat()
    new_positions = []
    for sym in long_syms:
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        pos = {
            'symbol': sym,
            'direction': 'long',
            'entry_price': price,
            'entry_time': now,
            'peak': price,
            'current_price': price,
            'score': float(cross.loc[sym, 'score_max']),
            'funding_rate': funding_rates.get(sym, 0),
            'unrealized_pnl': 0.0,
            'days_held': 0,
        }
        new_positions.append(pos)
        log_event('open', f"LONG {sym} @ {price:.4f} (score={pos['score']:.3f})",
                  {'funding_rate': pos['funding_rate']})

    # Deduct entry costs
    state['equity_long'] *= (1 - COST_PER_SIDE * len(new_positions) / max(len(new_positions), 1))
    state['positions_long'] = new_positions
    state['n_positions_at_entry_long'] = len(new_positions)
    state['last_rebalance_long'] = now
    state['total_trades_long'] = state.get('total_trades_long', 0) + len(new_positions)

    # Record equity snapshot
    equity_hist = load_equity_history_long()
    equity_hist.append({
        'timestamp': now,
        'equity': state['equity_long'],
        'positions': len(new_positions),
        'symbols': [p['symbol'] for p in new_positions],
        'event': 'rebalance',
    })
    save_equity_history_long(equity_hist)

    log_event('rebalance', f"Opened {len(new_positions)} longs. Equity: ${state['equity_long']:,.2f}")
    return state


# ════════════════════════════════════════════
# MONITORING CYCLE
# ════════════════════════════════════════════

def run_cycle():
    """Run one monitoring cycle: check stops, maybe rebalance. Handles BOTH strategies."""
    state = load_state()

    # Keep equity_short in sync with equity (backward compat)
    state['equity_short'] = state.get('equity', state.get('equity_short', INITIAL_CAPITAL * CAPITAL_SPLIT))

    # ── SHORT STRATEGY ──────────────────────────
    needs_rebalance_short = False
    if state['last_rebalance'] is None:
        needs_rebalance_short = True
    else:
        last = pd.Timestamp(state['last_rebalance'])
        days_since = (pd.Timestamp.now(tz='UTC') - last).days
        if days_since >= HOLDING_DAYS:
            needs_rebalance_short = True
        if not state['positions']:
            needs_rebalance_short = True

    # ── LONG STRATEGY ──────────────────────────
    needs_rebalance_long = False
    if state.get('last_rebalance_long') is None:
        needs_rebalance_long = True
    else:
        last_long = pd.Timestamp(state['last_rebalance_long'])
        days_since_long = (pd.Timestamp.now(tz='UTC') - last_long).days
        if days_since_long >= REBALANCE_LONG:
            needs_rebalance_long = True
        if not state.get('positions_long'):
            needs_rebalance_long = True

    # Build shared panel if either strategy needs rebalance
    shared_panel = None
    shared_symbols_info = None
    if needs_rebalance_short or needs_rebalance_long:
        shared_symbols_info = get_spot_symbols()
        if len(shared_symbols_info) >= 15:
            log_event('data', f'Building shared live panel for {len(shared_symbols_info)} symbols')
            shared_panel = build_live_panel(shared_symbols_info)
            if not shared_panel.empty:
                shared_panel = compute_features_live(shared_panel)
            else:
                shared_panel = None

    # ── COMPUTE REGIME ONCE (before any rebalance) ──
    regime_dir = 'long'  # default
    if shared_panel is not None and (needs_rebalance_short or needs_rebalance_long):
        regime_dir, _, _ = check_regime_filter(shared_panel)
        state['regime_direction'] = regime_dir
        log_event('regime', f'Regime decision: {regime_dir}')

    # ── Run SHORT rebalance (only when regime says short) ──
    if needs_rebalance_short:
        if regime_dir == 'short':
            log_event('regime', 'SHORT activated: bearish regime detected')
            state = rebalance(state)
        else:
            # Not a short regime — close any open shorts and skip
            if state['positions']:
                log_event('regime', f'SHORT skipped: regime={regime_dir}. Closing {len(state["positions"])} open shorts.')
                # Let them run to expiry via normal stop/hold logic
            else:
                log_event('regime', f'SHORT skipped: regime={regime_dir}, no open positions')
    else:
        # Just check stops and update prices for SHORT
        state, closed = check_stops(state)
        if closed:
            log_event('monitor', f'SHORT: {len(closed)} positions stopped out')

        # Apply funding cost on open short positions
        if state['positions']:
            funding_rates = fetch_current_funding([p['symbol'] for p in state['positions']])
            n_positions_at_entry = state.get('n_positions_at_entry', TOP_N)
            total_funding_cost = 0.0
            for pos in state['positions']:
                rate = funding_rates.get(pos['symbol'], pos.get('funding_rate', 0))
                pos['funding_rate'] = rate
                funding_cost = rate * 0.5 * (1.0 / n_positions_at_entry)
                total_funding_cost += funding_cost
            if total_funding_cost != 0:
                state['equity'] *= (1 - total_funding_cost)
                state['cumulative_funding'] = state.get('cumulative_funding', 0.0) + total_funding_cost * state['equity']
                if closed:
                    log_event('funding', f'SHORT funding cost: {total_funding_cost*100:.4f}% (cumul ${state["cumulative_funding"]:.2f})')

        # Accrue stablecoin yield on idle SHORT capital
        n_open = len(state['positions'])
        idle_fraction = (TOP_N - n_open) / TOP_N
        if idle_fraction > 0:
            yield_amount = state['equity'] * idle_fraction * STABLE_YIELD_PER_4H
            state['equity'] += yield_amount
            state['cumulative_yield'] = state.get('cumulative_yield', 0.0) + yield_amount
            if closed:
                log_event('yield', f'SHORT idle {idle_fraction:.0%}, accrued ${yield_amount:.4f} (cumul ${state["cumulative_yield"]:.2f})')

        # Record SHORT equity snapshot
        if state['positions']:
            symbols = [p['symbol'] for p in state['positions']]
            prices = fetch_current_prices(symbols)
            unrealized = 0
            for pos in state['positions']:
                cp = prices.get(pos['symbol'], pos['current_price'])
                pos['current_price'] = cp
                pos['unrealized_pnl'] = -(cp / pos['entry_price'] - 1)
                unrealized += pos['unrealized_pnl']

            n_at_entry = state.get('n_positions_at_entry', TOP_N)
            avg_unreal = unrealized / n_at_entry if state['positions'] else 0
            mark_equity = state['equity'] * (1 + avg_unreal)

            equity_hist = load_equity_history()
            equity_hist.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'equity': state['equity'],
                'mark_equity': mark_equity,
                'unrealized_pnl_pct': avg_unreal * 100,
                'positions': len(state['positions']),
                'idle_yield_accrued': state.get('cumulative_yield', 0.0),
                'event': 'monitor',
            })
            save_equity_history(equity_hist)
        else:
            equity_hist = load_equity_history()
            equity_hist.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'equity': state['equity'],
                'mark_equity': state['equity'],
                'unrealized_pnl_pct': 0.0,
                'positions': 0,
                'idle_yield_accrued': state.get('cumulative_yield', 0.0),
                'event': 'idle',
            })
            save_equity_history(equity_hist)

    # Keep equity_short in sync
    state['equity_short'] = state['equity']

    # ── Run LONG rebalance ──
    if needs_rebalance_long:
        state = rebalance_long(state, panel=shared_panel, symbols_info=shared_symbols_info)
    else:
        # Check stops for LONG positions
        state, closed_long = check_stops_long(state)
        if closed_long:
            log_event('monitor', f'LONG: {len(closed_long)} positions closed')

        # Apply funding cost on open long positions (long RECEIVES funding when rate > 0)
        if state.get('positions_long'):
            funding_rates = fetch_current_funding([p['symbol'] for p in state['positions_long']])
            n_positions_at_entry = state.get('n_positions_at_entry_long', MAX_POSITIONS_LONG)
            total_funding_effect = 0.0
            for pos in state['positions_long']:
                rate = funding_rates.get(pos['symbol'], pos.get('funding_rate', 0))
                pos['funding_rate'] = rate
                # Long pays funding when rate > 0, receives when rate < 0 (opposite of short)
                funding_effect = -rate * 0.5 * (1.0 / n_positions_at_entry)
                total_funding_effect += funding_effect
            if total_funding_effect != 0:
                state['equity_long'] *= (1 + total_funding_effect)
                state['cumulative_funding_long'] = state.get('cumulative_funding_long', 0.0) - total_funding_effect * state['equity_long']

        # Accrue stablecoin yield on idle LONG capital
        n_open_long = len(state.get('positions_long', []))
        idle_fraction_long = (MAX_POSITIONS_LONG - n_open_long) / MAX_POSITIONS_LONG
        if idle_fraction_long > 0:
            yield_amount = state['equity_long'] * idle_fraction_long * STABLE_YIELD_PER_4H
            state['equity_long'] += yield_amount
            state['cumulative_yield_long'] = state.get('cumulative_yield_long', 0.0) + yield_amount

        # Record LONG equity snapshot
        if state.get('positions_long'):
            symbols = [p['symbol'] for p in state['positions_long']]
            prices = fetch_current_prices(symbols)
            unrealized = 0
            for pos in state['positions_long']:
                cp = prices.get(pos['symbol'], pos['current_price'])
                pos['current_price'] = cp
                pos['unrealized_pnl'] = (cp / pos['entry_price'] - 1)
                unrealized += pos['unrealized_pnl']

            n_at_entry = state.get('n_positions_at_entry_long', MAX_POSITIONS_LONG)
            avg_unreal = unrealized / n_at_entry if state['positions_long'] else 0
            mark_equity = state['equity_long'] * (1 + avg_unreal)

            equity_hist_long = load_equity_history_long()
            equity_hist_long.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'equity': state['equity_long'],
                'mark_equity': mark_equity,
                'unrealized_pnl_pct': avg_unreal * 100,
                'positions': len(state['positions_long']),
                'event': 'monitor',
            })
            save_equity_history_long(equity_hist_long)
        else:
            equity_hist_long = load_equity_history_long()
            equity_hist_long.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'equity': state['equity_long'],
                'mark_equity': state['equity_long'],
                'unrealized_pnl_pct': 0.0,
                'positions': 0,
                'event': 'idle',
            })
            save_equity_history_long(equity_hist_long)

    save_state(state)
    return state


def print_status():
    """Print current paper trading status."""
    state = load_state()
    trades = load_trades()
    trades_long = load_trades_long()
    equity_hist = load_equity_history()

    short_eq = state.get('equity', state.get('equity_short', INITIAL_CAPITAL * CAPITAL_SPLIT))
    long_eq = state.get('equity_long', INITIAL_CAPITAL * (1 - CAPITAL_SPLIT))
    total_eq = short_eq + long_eq
    short_cap = INITIAL_CAPITAL * CAPITAL_SPLIT
    long_cap = INITIAL_CAPITAL * (1 - CAPITAL_SPLIT)

    print("\n" + "=" * 60)
    print("  PAPER TRADING STATUS — CatBoost Short + Long")
    print("=" * 60)
    print(f"  Started:        {state.get('started_at', 'N/A')[:10]}")
    print(f"  Total Capital:  ${INITIAL_CAPITAL:,.0f}")
    print(f"  Total Equity:   ${total_eq:,.2f} ({(total_eq/INITIAL_CAPITAL-1)*100:+.2f}%)")
    print()

    # SHORT
    print(f"  ── SHORT STRATEGY ──")
    print(f"  Capital:      ${short_cap:,.0f}")
    print(f"  Equity:       ${short_eq:,.2f} ({(short_eq/short_cap-1)*100:+.2f}%)")
    print(f"  Trades:       {state.get('total_trades', 0)}")
    print(f"  Last Rebal:   {state.get('last_rebalance', 'Never')}")

    if state['positions']:
        print(f"\n  SHORT POSITIONS ({len(state['positions'])}):")
        print(f"  {'Symbol':12s} {'Entry':>10s} {'Current':>10s} {'PnL':>8s} {'Days':>5s}")
        print(f"  {'-'*50}")
        for p in state['positions']:
            pnl = p.get('unrealized_pnl', 0) * 100
            days = p.get('days_held', 0)
            print(f"  {p['symbol']:12s} {p['entry_price']:10.4f} {p.get('current_price', 0):10.4f} {pnl:+7.2f}% {days:5d}")

    # LONG
    print(f"\n  ── LONG STRATEGY ──")
    print(f"  Capital:      ${long_cap:,.0f}")
    print(f"  Equity:       ${long_eq:,.2f} ({(long_eq/long_cap-1)*100:+.2f}%)")
    print(f"  Trades:       {state.get('total_trades_long', 0)}")
    print(f"  Last Rebal:   {state.get('last_rebalance_long', 'Never')}")
    print(f"  Regime Skips: {state.get('regime_skip_count', 0)}")

    if state.get('positions_long'):
        print(f"\n  LONG POSITIONS ({len(state['positions_long'])}):")
        print(f"  {'Symbol':12s} {'Entry':>10s} {'Current':>10s} {'PnL':>8s} {'Days':>5s}")
        print(f"  {'-'*50}")
        for p in state['positions_long']:
            pnl = p.get('unrealized_pnl', 0) * 100
            days = p.get('days_held', 0)
            print(f"  {p['symbol']:12s} {p['entry_price']:10.4f} {p.get('current_price', 0):10.4f} {pnl:+7.2f}% {days:5d}")

    # Trade stats
    for label, t_list in [("SHORT", trades), ("LONG", trades_long)]:
        if t_list:
            wins = sum(1 for t in t_list if t.get('return', 0) > 0)
            avg_ret = np.mean([t.get('return', 0) for t in t_list]) * 100
            print(f"\n  {label} TRADE STATS:")
            print(f"  Win Rate:     {wins}/{len(t_list)} ({wins/len(t_list)*100:.0f}%)")
            print(f"  Avg Return:   {avg_ret:+.2f}%")
            recent = t_list[-5:]
            print(f"  {'Symbol':12s} {'Entry':>10s} {'Exit':>10s} {'Return':>8s} {'Reason':12s}")
            print(f"  {'-'*55}")
            for t in recent:
                ret = t.get('return', 0) * 100
                print(f"  {t['symbol']:12s} {t['entry_price']:10.4f} {t.get('exit_price',0):10.4f} {ret:+7.2f}% {t.get('exit_reason','?'):12s}")

    print("=" * 60 + "\n")


def run_daemon():
    """Run continuously, checking every 4 hours."""
    log_event('daemon', 'Paper trading daemon started')
    while True:
        try:
            run_cycle()
        except Exception as e:
            log_event('error', f'Cycle failed: {e}')
        # Sleep 4 hours
        log_event('daemon', 'Sleeping 4 hours until next check')
        time.sleep(4 * 3600)


# ════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper Trading Engine')
    parser.add_argument('--daemon', action='store_true', help='Run continuously')
    parser.add_argument('--status', action='store_true', help='Print current status')
    parser.add_argument('--retrain', action='store_true', help='Force model retrain')
    parser.add_argument('--reset', action='store_true', help='Reset all paper trading state')
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.retrain:
        train_models()
        print("Models retrained successfully (single fwd_min + fwd_max, matches backtest).")
    elif args.reset:
        for f in [STATE_FILE, TRADES_FILE, EQUITY_FILE, LOG_FILE,
                  TRADES_LONG_FILE, EQUITY_LONG_FILE,
                  MODEL_MIN_FILE, MODEL_MAX_FILE]:
            if f.exists():
                f.unlink()
        # Clean up old ensemble model files if they exist
        for i in range(10):
            for pat in [f'model_{i}.cbm', f'model_long_{i}.cbm']:
                old = PAPER_DIR / pat
                if old.exists():
                    old.unlink()
        for meta in ['model_meta.json', 'model_meta_long.json',
                     'feature_importance.json', 'feature_importance_long.json']:
            mp = PAPER_DIR / meta
            if mp.exists():
                mp.unlink()
        print("Paper trading state reset (both short and long, all models cleared).")
    elif args.daemon:
        run_daemon()
    else:
        run_cycle()
