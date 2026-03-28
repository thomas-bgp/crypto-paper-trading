"""
Paper Trading Engine — CatBoost Short-Only Strategy
Runs the ML model against LIVE Binance data without real capital.
Trains on historical data, then monitors real positions every rebalance cycle.

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

# ─── Config (mirrors backtest exactly) ───
HOLDING_DAYS = 5               # Changed from 14: signal decays fast, 5d captures more alpha
TOP_N = 5
UNIVERSE_TOP = 50
COST_PER_SIDE = 0.002      # 0.20% simulated cost
STOP_PCT = 0.15             # 15% trailing stop
INITIAL_CAPITAL = 10_000    # paper capital
TRAIN_MONTHS = 18
N_ENSEMBLE = 3
STABLE_APY = 0.05              # 5% annual stablecoin yield on idle capital
STABLE_YIELD_PER_4H = STABLE_APY / 365 / 6  # yield per 4h cycle

BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_TICKER = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_MARK_PRICE = "https://fapi.binance.com/fapi/v1/premiumIndex"

CORE_FEATURES = [
    'mom_14', 'mom_28', 'mom_56', 'mom_14_skip1',
    'poly_slope_28', 'poly_curve_28', 'poly_slope_56', 'poly_curve_56',
    'rvol_28', 'vol_ratio', 'max_ret_28', 'min_ret_28',
    'amihud', 'spread_28', 'turnover_28',
    'rsi_14', 'macd_hist', 'donchian_pos',
    'mom_14_csrank', 'rvol_28_csrank',
]

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
MODEL_FILE = PAPER_DIR / 'model_ensemble.cbm'
LOG_FILE = PAPER_DIR / 'log.jsonl'


# ════════════════════════════════════════════
# BINANCE DATA (LIVE)
# ════════════════════════════════════════════

def get_futures_symbols(min_volume_usd=5_000_000):
    """Get actively traded USDT-M futures sorted by volume."""
    try:
        resp = requests.get(BINANCE_TICKER, timeout=15)
        tickers = resp.json()
        result = []
        for t in tickers:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT') or '_' in sym:
                continue
            vol = float(t.get('quoteVolume', 0))
            if vol >= min_volume_usd:
                result.append({'symbol': sym, 'volume_24h': vol, 'price': float(t.get('lastPrice', 0))})
        return sorted(result, key=lambda x: -x['volume_24h'])[:UNIVERSE_TOP]
    except Exception as e:
        log_event('error', f'Failed to get futures symbols: {e}')
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
    """Get current mark prices for a list of symbols."""
    try:
        resp = requests.get(BINANCE_MARK_PRICE, timeout=15)
        data = resp.json()
        prices = {}
        for item in data:
            if item['symbol'] in symbols:
                prices[item['symbol']] = float(item['markPrice'])
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
    """Compute the 20 core features on live panel (same logic as ml_features.py)."""
    g = panel.groupby(level='symbol')
    close = panel['close']
    high = panel['high']
    low = panel['low']
    qvol = panel['quote_vol']

    panel['log_ret'] = g['close'].transform(lambda x: np.log(x / x.shift(1)))

    # Momentum
    for lb in [14, 28, 56]:
        panel[f'mom_{lb}'] = g['close'].pct_change(lb)
    panel['mom_14_skip1'] = g['close'].shift(1).transform(lambda x: x.pct_change(14))

    # Volatility
    for w in [14, 28, 56]:
        panel[f'rvol_{w}'] = g['log_ret'].transform(lambda x: x.rolling(w).std() * np.sqrt(365))
    panel['vol_ratio'] = panel['rvol_14'] / (panel['rvol_56'] + 1e-10)
    panel['max_ret_28'] = g['log_ret'].transform(lambda x: x.rolling(28).max())
    panel['min_ret_28'] = g['log_ret'].transform(lambda x: x.rolling(28).min())

    # Liquidity
    panel['vol_avg_28'] = g['quote_vol'].transform(lambda x: x.rolling(28).mean())
    panel['turnover'] = qvol / (close * 1e6 + 1)
    panel['turnover_28'] = g['turnover'].transform(lambda x: x.rolling(28).mean())
    panel['amihud'] = (panel['log_ret'].abs() / (qvol + 1)).groupby(level='symbol').transform(
        lambda x: x.rolling(28).mean()) * 1e9
    panel['spread'] = 2 * (high - low) / (high + low + 1e-10)
    panel['spread_28'] = g['spread'].transform(lambda x: x.rolling(28).mean())

    # Technical
    delta = g['close'].diff()
    gain = delta.clip(lower=0)
    loss_s = (-delta).clip(lower=0)
    ag = gain.groupby(level='symbol').transform(lambda x: x.ewm(span=14, adjust=False).mean())
    al = loss_s.groupby(level='symbol').transform(lambda x: x.ewm(span=14, adjust=False).mean())
    panel['rsi_14'] = 100 - 100 / (1 + ag / (al + 1e-10))

    ema12 = g['close'].transform(lambda x: x.ewm(span=12).mean())
    ema26 = g['close'].transform(lambda x: x.ewm(span=26).mean())
    macd = ema12 - ema26
    signal = macd.groupby(level='symbol').transform(lambda x: x.ewm(span=9).mean())
    panel['macd_hist'] = (macd - signal) / (close + 1e-10)

    h20 = g['high'].transform(lambda x: x.rolling(20).max())
    l20 = g['low'].transform(lambda x: x.rolling(20).min())
    panel['donchian_pos'] = (close - l20) / (h20 - l20 + 1e-10)

    # Polynomial derivatives
    for window in [28, 56]:
        _compute_poly(panel, window)

    # Cross-sectional
    panel['mom_14_csrank'] = panel.groupby(level='date')['mom_14'].rank(pct=True)
    panel['rvol_28_csrank'] = panel.groupby(level='date')['rvol_28'].rank(pct=True)

    return panel


def _compute_poly(panel, window):
    """Compute polynomial slope and curvature."""
    results = {f'poly_slope_{window}': [], f'poly_curve_{window}': []}
    x = np.arange(window, dtype=np.float64)
    x_norm = (x - x.mean()) / (x.std() + 1e-10)
    X = np.column_stack([x_norm**2, x_norm, np.ones(window)])
    XtX_inv_Xt = np.linalg.pinv(X)

    for sym, grp in panel.groupby(level='symbol'):
        c = np.log(grp['close'].values + 1e-10)
        n = len(c)
        slope = np.full(n, np.nan)
        curve = np.full(n, np.nan)
        for i in range(window, n):
            y = c[i-window:i]
            if np.any(np.isnan(y)):
                continue
            beta = XtX_inv_Xt @ y
            slope[i] = beta[1]
            curve[i] = beta[0]
        idx = grp.index
        results[f'poly_slope_{window}'].append(pd.Series(slope, index=idx))
        results[f'poly_curve_{window}'].append(pd.Series(curve, index=idx))

    for col, series_list in results.items():
        if series_list:
            panel[col] = pd.concat(series_list)


# ════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════

def train_model():
    """Train CatBoost ensemble on historical data (same as backtest)."""
    from catboost import CatBoostRanker

    log_event('train', 'Starting model training on historical data')

    # Use the existing ml_features module
    sys.path.insert(0, str(SCRIPT_DIR))
    from ml_features import load_daily_panel, compute_all_features

    panel = load_daily_panel()
    panel = compute_all_features(panel)

    feat_cols = [f for f in CORE_FEATURES if f in panel.columns]
    dates = panel.index.get_level_values('date').unique().sort_values()
    train_end = dates[-1]
    train_start = train_end - pd.DateOffset(months=TRAIN_MONTHS)

    train_mask = ((panel.index.get_level_values('date') >= train_start) &
                  (panel.index.get_level_values('date') <= train_end))
    train = panel[train_mask].copy()
    g = train.groupby(level='symbol')
    train['raw_fwd'] = g['close'].transform(lambda x: x.pct_change(HOLDING_DAYS).shift(-HOLDING_DAYS))
    market_ret = train.groupby(level='date')['raw_fwd'].transform('mean')
    train['neutral_fwd'] = train['raw_fwd'] - market_ret

    purge_cutoff = train.index.get_level_values('date').max() - pd.Timedelta(days=HOLDING_DAYS + 4)
    train = train[train.index.get_level_values('date') <= purge_cutoff]
    train = train.dropna(subset=['neutral_fwd'])

    p1, p99 = train['neutral_fwd'].quantile([0.02, 0.98])
    train['neutral_fwd'] = train['neutral_fwd'].clip(p1, p99)
    train['target_rank'] = train.groupby(level='date')['neutral_fwd'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') if len(x) >= 5 else 2
    ).fillna(2).astype(int)

    if 'vol_avg_28' in panel.columns:
        vol = panel.loc[train.index, 'vol_avg_28']
        thresh = vol.groupby(level='date').transform(lambda x: x.quantile(0.3))
        train = train[vol > thresh]

    # Train ensemble
    models = []
    n_dates = train.index.get_level_values('date').nunique()
    train_dates = train.index.get_level_values('date').unique().sort_values()

    for k in range(N_ENSEMBLE):
        offset = int(n_dates * 0.2 * k / N_ENSEMBLE)
        end_idx = n_dates - int(n_dates * 0.2 * (N_ENSEMBLE - 1 - k) / N_ENSEMBLE)
        subset_dates = train_dates[offset:end_idx]
        subset = train[train.index.get_level_values('date').isin(subset_dates)]

        if len(subset) < 500:
            continue

        X = np.nan_to_num(subset[feat_cols].values, nan=0, posinf=0, neginf=0)
        y = subset['target_rank'].values
        gids = pd.Categorical(subset.index.get_level_values('date')).codes

        params = CATBOOST_PARAMS.copy()
        params['random_seed'] = 42 + k
        model = CatBoostRanker(**params)
        model.fit(X, y, group_id=gids)
        models.append(model)

    # Save models
    for i, m in enumerate(models):
        m.save_model(str(PAPER_DIR / f'model_{i}.cbm'))

    # Save feature importance
    if models:
        imp = pd.Series(
            models[0].get_feature_importance(type='PredictionValuesChange'),
            index=feat_cols
        ).sort_values(ascending=False)
        imp_dict = (imp / (imp.sum() + 1e-10)).to_dict()
        with open(PAPER_DIR / 'feature_importance.json', 'w') as f:
            json.dump(imp_dict, f, indent=2)

    meta = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'train_start': str(train_start.date()),
        'train_end': str(train_end.date()),
        'n_models': len(models),
        'n_rows': len(train),
        'features': feat_cols,
    }
    with open(PAPER_DIR / 'model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    log_event('train', f'Trained {len(models)} models on {len(train)} rows ({train_start.date()} to {train_end.date()})')
    return models, feat_cols


def load_models():
    """Load saved models."""
    from catboost import CatBoostRanker
    models = []
    for i in range(N_ENSEMBLE):
        path = PAPER_DIR / f'model_{i}.cbm'
        if path.exists():
            m = CatBoostRanker()
            m.load_model(str(path))
            models.append(m)
    meta_path = PAPER_DIR / 'model_meta.json'
    feat_cols = CORE_FEATURES
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            feat_cols = meta.get('features', CORE_FEATURES)
    return models, feat_cols


def models_need_retrain():
    """Check if models are stale (>56 days old)."""
    meta_path = PAPER_DIR / 'model_meta.json'
    if not meta_path.exists():
        return True
    with open(meta_path) as f:
        meta = json.load(f)
    trained = pd.Timestamp(meta['trained_at'])
    age_days = (pd.Timestamp.now(tz='UTC') - trained).days
    return age_days >= 56


# ════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════

def load_state():
    """Load paper trading state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'capital': INITIAL_CAPITAL,
        'equity': INITIAL_CAPITAL,
        'positions': [],
        'last_rebalance': None,
        'total_trades': 0,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'cumulative_yield': 0.0,
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
    """Run the full rebalance cycle: close old positions, predict, open new."""
    log_event('rebalance', 'Starting rebalance cycle')

    # Load or train models
    models, feat_cols = load_models()
    if not models or models_need_retrain():
        log_event('rebalance', 'Models need training')
        models, feat_cols = train_model()

    if not models:
        log_event('error', 'No models available, skipping rebalance')
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

    # Get live universe
    symbols_info = get_futures_symbols()
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

    # Ensemble prediction
    scores = []
    for m in models:
        try:
            s = m.predict(X)
            s = (s - s.mean()) / (s.std() + 1e-10)
            scores.append(s)
        except Exception as e:
            log_event('error', f'Model predict failed: {e}')

    if not scores:
        log_event('error', 'All model predictions failed')
        return state

    cross = cross.copy()
    cross['score'] = np.mean(scores, axis=0)
    cross = cross.sort_values('score', ascending=True)  # lowest score = worst = short

    # Select bottom N to short
    short_syms = cross.head(TOP_N).index.tolist()

    # Get entry prices
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
            'score': float(cross.loc[sym, 'score']),
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
# MONITORING CYCLE
# ════════════════════════════════════════════

def run_cycle():
    """Run one monitoring cycle: check stops, maybe rebalance."""
    state = load_state()

    # Check if we need to rebalance
    needs_rebalance = False
    if state['last_rebalance'] is None:
        needs_rebalance = True
    else:
        last = pd.Timestamp(state['last_rebalance'])
        days_since = (pd.Timestamp.now(tz='UTC') - last).days
        if days_since >= HOLDING_DAYS:
            needs_rebalance = True
        # Also rebalance if all positions were stopped out
        if not state['positions']:
            needs_rebalance = True

    if needs_rebalance:
        state = rebalance(state)
    else:
        # Just check stops and update prices
        state, closed = check_stops(state)
        if closed:
            log_event('monitor', f'{len(closed)} positions stopped out')

        # Apply funding cost on open short positions (Binance: 3x/day, our cycle: 4h = 0.5 funding periods)
        if state['positions']:
            funding_rates = fetch_current_funding([p['symbol'] for p in state['positions']])
            n_positions_at_entry = state.get('n_positions_at_entry', TOP_N)
            total_funding_cost = 0.0
            for pos in state['positions']:
                rate = funding_rates.get(pos['symbol'], pos.get('funding_rate', 0))
                pos['funding_rate'] = rate
                # Short pays funding when rate > 0, receives when rate < 0
                # Each position has weight 1/n_positions_at_entry of equity
                # Funding is applied per 8h; our cycle is 4h = 0.5 funding periods
                funding_cost = rate * 0.5 * (1.0 / n_positions_at_entry)
                total_funding_cost += funding_cost
            if total_funding_cost != 0:
                state['equity'] *= (1 - total_funding_cost)
                state['cumulative_funding'] = state.get('cumulative_funding', 0.0) + total_funding_cost * state['equity']
                if closed:
                    log_event('funding', f'Funding cost: {total_funding_cost*100:.4f}% (cumul ${state["cumulative_funding"]:.2f})')

        # Accrue stablecoin yield on idle capital
        n_open = len(state['positions'])
        idle_fraction = (TOP_N - n_open) / TOP_N
        if idle_fraction > 0:
            yield_amount = state['equity'] * idle_fraction * STABLE_YIELD_PER_4H
            state['equity'] += yield_amount
            state['cumulative_yield'] = state.get('cumulative_yield', 0.0) + yield_amount
            if closed:  # only log when something changed, avoid spamming every 4h
                log_event('yield', f'Idle {idle_fraction:.0%} of capital, accrued ${yield_amount:.4f} (cumul ${state["cumulative_yield"]:.2f})')

        # Record equity snapshot with current unrealized PnL
        if state['positions']:
            symbols = [p['symbol'] for p in state['positions']]
            prices = fetch_current_prices(symbols)
            unrealized = 0
            for pos in state['positions']:
                cp = prices.get(pos['symbol'], pos['current_price'])
                pos['current_price'] = cp
                pos['unrealized_pnl'] = -(cp / pos['entry_price'] - 1)
                unrealized += pos['unrealized_pnl']

            avg_unreal = unrealized / len(state['positions']) if state['positions'] else 0
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
            # All positions closed, still accrue and record
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

    save_state(state)
    return state


def print_status():
    """Print current paper trading status."""
    state = load_state()
    trades = load_trades()
    equity_hist = load_equity_history()

    print("\n" + "=" * 60)
    print("  PAPER TRADING STATUS — CatBoost Short-Only")
    print("=" * 60)
    print(f"  Started:      {state.get('started_at', 'N/A')[:10]}")
    print(f"  Capital:      ${INITIAL_CAPITAL:,.0f}")
    print(f"  Equity:       ${state['equity']:,.2f}")
    ret = (state['equity'] / INITIAL_CAPITAL - 1) * 100
    print(f"  Return:       {ret:+.2f}%")
    print(f"  Total Trades: {state['total_trades']}")
    cum_yield = state.get('cumulative_yield', 0.0)
    if cum_yield > 0:
        print(f"  Stable Yield: ${cum_yield:,.2f} ({cum_yield/INITIAL_CAPITAL*100:.2f}%)")
    idle_frac = (TOP_N - len(state.get('positions', []))) / TOP_N
    if idle_frac > 0:
        print(f"  Idle Capital: {idle_frac:.0%} → earning {STABLE_APY*100:.1f}% APY")
    print(f"  Last Rebal:   {state.get('last_rebalance', 'Never')}")

    if state['positions']:
        print(f"\n  OPEN POSITIONS ({len(state['positions'])}):")
        print(f"  {'Symbol':12s} {'Entry':>10s} {'Current':>10s} {'PnL':>8s} {'Days':>5s}")
        print(f"  {'-'*50}")
        for p in state['positions']:
            pnl = p.get('unrealized_pnl', 0) * 100
            days = p.get('days_held', 0)
            print(f"  {p['symbol']:12s} {p['entry_price']:10.4f} {p.get('current_price', 0):10.4f} {pnl:+7.2f}% {days:5d}")

    if trades:
        # Recent trades
        recent = trades[-10:]
        wins = sum(1 for t in trades if t.get('return', 0) > 0)
        avg_ret = np.mean([t.get('return', 0) for t in trades]) * 100 if trades else 0

        print(f"\n  TRADE STATS:")
        print(f"  Win Rate:     {wins}/{len(trades)} ({wins/len(trades)*100:.0f}%)")
        print(f"  Avg Return:   {avg_ret:+.2f}%")

        print(f"\n  RECENT TRADES:")
        print(f"  {'Symbol':12s} {'Entry':>10s} {'Exit':>10s} {'Return':>8s} {'Reason':12s}")
        print(f"  {'-'*55}")
        for t in recent:
            ret = t.get('return', 0) * 100
            print(f"  {t['symbol']:12s} {t['entry_price']:10.4f} {t.get('exit_price',0):10.4f} {ret:+7.2f}% {t.get('exit_reason','?'):12s}")

    if equity_hist:
        n = len(equity_hist)
        if n >= 2:
            first = equity_hist[0]['equity']
            last_eq = equity_hist[-1].get('mark_equity', equity_hist[-1]['equity'])
            total_ret = (last_eq / first - 1) * 100
            print(f"\n  EQUITY HISTORY: {n} snapshots, total return: {total_ret:+.2f}%")

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
        train_model()
        print("Model retrained successfully.")
    elif args.reset:
        for f in [STATE_FILE, TRADES_FILE, EQUITY_FILE, LOG_FILE]:
            if f.exists():
                f.unlink()
        print("Paper trading state reset.")
    elif args.daemon:
        run_daemon()
    else:
        run_cycle()
