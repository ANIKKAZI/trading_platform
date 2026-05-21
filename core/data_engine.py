"""
Data ingestion and caching layer.
Fetches OHLCV data for all symbols using yfinance.
"""

import os
import logging
import datetime
import pandas as pd
import yfinance as yf

from config.settings import (
    HISTORICAL_YEARS, DATA_CACHE_DIR, BENCHMARK_SYMBOL
)

logger = logging.getLogger(__name__)


def _cache_path(symbol: str) -> str:
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    return os.path.join(DATA_CACHE_DIR, f"{symbol}.parquet")


def fetch_symbol(symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Return OHLCV DataFrame for a single symbol.
    Uses local parquet cache; refreshes if today's data is missing or force_refresh=True.
    """
    cache = _cache_path(symbol)
    end = datetime.date.today()
    start = end - datetime.timedelta(days=int(HISTORICAL_YEARS * 365.25))

    if not force_refresh and os.path.exists(cache):
        df = pd.read_parquet(cache)
        last_date = df.index.max().date()
        if last_date >= end - datetime.timedelta(days=3):
            logger.debug("Cache hit for %s (last=%s)", symbol, last_date)
            return df
        logger.debug("Stale cache for %s, refreshing", symbol)

    logger.info("Downloading %s  [%s → %s]", symbol, start, end)
    raw = yf.download(symbol, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {symbol}")

    raw.index = pd.to_datetime(raw.index)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    raw.to_parquet(cache)
    return raw


def fetch_universe(symbols: list[str], force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch data for all symbols, including benchmark."""
    data: dict[str, pd.DataFrame] = {}
    all_symbols = list(set(symbols + [BENCHMARK_SYMBOL]))
    for sym in all_symbols:
        try:
            data[sym] = fetch_symbol(sym, force_refresh=force_refresh)
        except Exception as exc:
            logger.warning("Skipping %s: %s", sym, exc)
    return data
