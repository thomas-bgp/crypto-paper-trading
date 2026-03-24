"""
Data fetcher module — downloads historical OHLCV data from Binance.
"""
import requests
import pandas as pd
import time
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_klines(symbol: str, interval: str = '4h', start_date: str = '2022-01-01',
                 end_date: str = '2026-03-15') -> pd.DataFrame:
    """Fetch OHLCV klines from Binance Futures."""
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)
    all_data = []
    current = start_ms

    while current < end_ms:
        params = {
            'symbol': symbol, 'interval': interval,
            'startTime': current, 'endTime': end_ms, 'limit': 1500
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        data = resp.json()
        if not data or isinstance(data, dict):
            break
        all_data.extend(data)
        current = data[-1][0] + 1
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_vol',
        'taker_buy_quote_vol', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df[~df.index.duplicated(keep='first')]
    return df


def fetch_funding_rates(symbol: str = 'BTCUSDT', start_date: str = '2022-01-01') -> pd.DataFrame:
    """Fetch historical funding rates from Binance Futures."""
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now().timestamp() * 1000)
    all_data = []
    current = start_ms

    while current < end_ms:
        params = {
            'symbol': symbol, 'startTime': current,
            'endTime': end_ms, 'limit': 1000
        }
        resp = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
        data = resp.json()
        if not data or isinstance(data, dict):
            break
        all_data.extend(data)
        current = data[-1]['fundingTime'] + 1
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    df.set_index('fundingTime', inplace=True)
    return df[['fundingRate']]


def fetch_fear_greed(limit: int = 1500) -> pd.DataFrame:
    """Fetch Fear & Greed Index from alternative.me."""
    url = f"https://api.alternative.me/fng/?limit={limit}"
    resp = requests.get(url, timeout=30)
    data = resp.json().get('data', [])
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    df['value'] = df['value'].astype(int)
    df.set_index('timestamp', inplace=True)
    df = df[['value']].rename(columns={'value': 'fng'})
    return df.sort_index()


def download_all(symbols: list = None, start_date: str = '2022-01-01'):
    """Download and cache all data needed for backtesting."""
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
            'LINKUSDT', 'LTCUSDT', 'ATOMUSDT', 'NEARUSDT', 'ARBUSDT',
            'OPUSDT', 'APTUSDT', 'SUIUSDT', 'INJUSDT', 'TIAUSDT'
        ]

    # Download 4h klines
    for sym in symbols:
        path = os.path.join(DATA_DIR, f'{sym}_4h.parquet')
        if os.path.exists(path):
            print(f"  [SKIP] {sym} 4h already cached")
            continue
        print(f"  [DL] {sym} 4h klines...", end=' ')
        df = fetch_klines(sym, '4h', start_date)
        if not df.empty:
            df.to_parquet(path)
            print(f"OK ({len(df)} candles)")
        else:
            print("EMPTY")
        time.sleep(0.5)

    # Download BTC funding rates
    fr_path = os.path.join(DATA_DIR, 'BTCUSDT_funding.parquet')
    if not os.path.exists(fr_path):
        print("  [DL] BTC funding rates...", end=' ')
        df_fr = fetch_funding_rates('BTCUSDT', start_date)
        if not df_fr.empty:
            df_fr.to_parquet(fr_path)
            print(f"OK ({len(df_fr)} rates)")
    else:
        print("  [SKIP] BTC funding already cached")

    # Download Fear & Greed
    fng_path = os.path.join(DATA_DIR, 'fear_greed.parquet')
    if not os.path.exists(fng_path):
        print("  [DL] Fear & Greed Index...", end=' ')
        df_fng = fetch_fear_greed()
        if not df_fng.empty:
            df_fng.to_parquet(fng_path)
            print(f"OK ({len(df_fng)} days)")
    else:
        print("  [SKIP] Fear & Greed already cached")

    print("\n  Data download complete.")


if __name__ == '__main__':
    download_all()
