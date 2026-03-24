"""
Universe Fetcher — downloads ALL Binance USDT-M perpetual pairs.
Survivorship-bias-free: includes coins that existed at any point.
Stores data in data/universe/ as individual parquet files.
"""
import requests
import pandas as pd
import time
import os
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
UNIVERSE_DIR = os.path.join(DATA_DIR, 'universe')
os.makedirs(UNIVERSE_DIR, exist_ok=True)

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"

# Exclusions: stablecoins, leveraged, wrapped, index tokens
EXCLUDE_PATTERNS = [
    'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'FDUSDUSDT', 'DAIUSDT',
    'EURUSDT', 'GBPUSDT',  # fiat pairs
]
EXCLUDE_SUFFIXES = ['DOWNUSDT', 'UPUSDT', 'BULLUSDT', 'BEARUSDT']


def get_all_usdt_perps():
    """Get all USDT-M perpetual symbols from Binance Futures."""
    resp = requests.get(BINANCE_EXCHANGE_INFO, timeout=30)
    data = resp.json()
    symbols = []
    for s in data['symbols']:
        if (s['quoteAsset'] == 'USDT'
                and s['contractType'] == 'PERPETUAL'
                and s['status'] in ('TRADING', 'BREAK', 'CLOSE')):
            sym = s['symbol']
            if sym in EXCLUDE_PATTERNS:
                continue
            if any(sym.endswith(suf) for suf in EXCLUDE_SUFFIXES):
                continue
            symbols.append({
                'symbol': sym,
                'status': s['status'],
                'onboardDate': s.get('onboardDate', 0),
                'baseAsset': s['baseAsset'],
            })
    return symbols


def fetch_klines_safe(symbol, interval='4h', start_date='2020-01-01',
                      end_date='2026-03-16'):
    """Fetch klines with retry logic."""
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)
    all_data = []
    current = start_ms
    retries = 0

    while current < end_ms:
        params = {
            'symbol': symbol, 'interval': interval,
            'startTime': current, 'endTime': end_ms, 'limit': 1500
        }
        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(10)
                retries += 1
                if retries > 5:
                    break
                continue
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data or isinstance(data, dict):
                break
            all_data.extend(data)
            current = data[-1][0] + 1
            retries = 0
            time.sleep(0.15)
        except Exception:
            retries += 1
            if retries > 3:
                break
            time.sleep(2)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_vol',
        'taker_buy_quote_vol', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_vol']:
        df[col] = df[col].astype(float)
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume', 'quote_vol']]
    df = df[~df.index.duplicated(keep='first')]
    return df


def download_universe(start_date='2020-01-01', force=False):
    """Download 4h klines for ALL USDT perpetuals."""
    print("Fetching Binance USDT-M perpetual list...")
    perps = get_all_usdt_perps()
    print(f"Found {len(perps)} perpetuals (excl stables/leveraged)")

    # Save symbol metadata
    meta_path = os.path.join(UNIVERSE_DIR, '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(perps, f, indent=2, default=str)

    downloaded = 0
    skipped = 0
    failed = 0

    for i, info in enumerate(perps):
        sym = info['symbol']
        path = os.path.join(UNIVERSE_DIR, f'{sym}_4h.parquet')

        if os.path.exists(path) and not force:
            skipped += 1
            continue

        try:
            print(f"  [{i+1}/{len(perps)}] {sym}...", end=' ', flush=True)
        except UnicodeEncodeError:
            print(f"  [{i+1}/{len(perps)}] [unicode]...", end=' ', flush=True)
        df = fetch_klines_safe(sym, '4h', start_date)

        if not df.empty and len(df) >= 100:  # min 100 candles (~17 days)
            df.to_parquet(path)
            print(f"OK ({len(df)} candles, {df.index[0].date()} to {df.index[-1].date()})")
            downloaded += 1
        elif not df.empty:
            print(f"TOO SHORT ({len(df)} candles, skip)")
            failed += 1
        else:
            print("EMPTY")
            failed += 1

        # Rate limit management
        if (i + 1) % 50 == 0:
            print(f"  --- Progress: {i+1}/{len(perps)} | DL: {downloaded} | Skip: {skipped} | Fail: {failed} ---")
            time.sleep(2)

    print(f"\n=== UNIVERSE DOWNLOAD COMPLETE ===")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed/too short: {failed}")
    print(f"  Total symbols: {len(perps)}")
    return perps


def build_point_in_time_universe(universe_dir=None, min_candles=180):
    """
    Build survivorship-bias-free universe snapshots.
    For each month, determines which coins were active and liquid.
    Returns dict: {date: [list of eligible symbols]}
    """
    if universe_dir is None:
        universe_dir = UNIVERSE_DIR

    # Load all data
    all_data = {}
    for f in os.listdir(universe_dir):
        if f.endswith('_4h.parquet') and not f.startswith('_'):
            sym = f.replace('_4h.parquet', '')
            try:
                df = pd.read_parquet(os.path.join(universe_dir, f))
                if len(df) >= min_candles:
                    all_data[sym] = df
            except Exception:
                continue

    print(f"Loaded {len(all_data)} symbols with >= {min_candles} candles")

    # Build monthly snapshots
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    if not all_dates:
        return {}, all_data

    # Monthly rebalance dates (first trading day of each month)
    monthly_dates = pd.date_range(
        start=all_dates[0] + pd.DateOffset(months=6),  # need 6mo history
        end=all_dates[-1],
        freq='MS'
    )

    universe_snapshots = {}
    for date in monthly_dates:
        eligible = []
        for sym, df in all_data.items():
            # Must have data up to this date
            if df.index[0] > date - pd.DateOffset(days=180):
                continue  # less than 180 days of history
            if df.index[-1] < date:
                continue  # delisted before this date

            # Get recent data
            recent = df[df.index <= date].tail(180)  # last 180 candles (~30 days)
            if len(recent) < 90:
                continue

            # Liquidity filter: avg daily quote volume > $5M
            # 6 candles per day, use quote_vol
            if 'quote_vol' in recent.columns:
                daily_vol = recent['quote_vol'].tail(42).mean() * 6  # 7 days avg
            else:
                daily_vol = recent['volume'].tail(42).mean() * recent['close'].tail(42).mean() * 6

            if daily_vol < 5_000_000:  # $5M minimum
                continue

            eligible.append({
                'symbol': sym,
                'daily_vol': daily_vol,
                'close': recent['close'].iloc[-1],
                'candles': len(recent),
            })

        # Sort by volume (proxy for market cap), take top 80
        eligible.sort(key=lambda x: x['daily_vol'], reverse=True)
        universe_snapshots[date] = eligible[:80]

    print(f"Built {len(universe_snapshots)} monthly snapshots")
    for date in list(universe_snapshots.keys())[:3] + list(universe_snapshots.keys())[-3:]:
        n = len(universe_snapshots[date])
        print(f"  {date.date()}: {n} eligible coins")

    return universe_snapshots, all_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--build':
        snapshots, data = build_point_in_time_universe()
        # Save snapshots
        snap_path = os.path.join(UNIVERSE_DIR, '_snapshots.json')
        serializable = {str(k.date()): [s['symbol'] for s in v] for k, v in snapshots.items()}
        with open(snap_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved to {snap_path}")
    else:
        download_universe()
