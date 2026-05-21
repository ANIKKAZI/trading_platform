"""
Feature engineering pipeline.
Computes all technical indicators and market features in-place on OHLCV DataFrames.
"""

import numpy as np
import pandas as pd

from config.settings import (
    EMA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD, VOLUME_MA_PERIOD,
    MOMENTUM_PERIODS, VOLATILITY_WINDOW, BENCHMARK_SYMBOL
)


# ---------------------------------------------------------------------------
# Low-level indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Accepts a raw OHLCV DataFrame and returns an enriched copy with all features.
    benchmark: SPY OHLCV DataFrame for beta computation.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- EMAs ---
    for span in EMA_PERIODS:
        df[f"EMA_{span}"] = _ema(close, span)

    # --- RSI ---
    df["RSI"] = _rsi(close, RSI_PERIOD)

    # --- MACD ---
    ema_fast = _ema(close, MACD_FAST)
    ema_slow = _ema(close, MACD_SLOW)
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = _ema(df["MACD"], MACD_SIGNAL)
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # --- ATR ---
    df["ATR"] = _atr(high, low, close, ATR_PERIOD)
    df["ATR_pct"] = df["ATR"] / close

    # --- Bollinger Bands ---
    bb_mid = close.rolling(BOLLINGER_PERIOD).mean()
    bb_std = close.rolling(BOLLINGER_PERIOD).std()
    df["BB_upper"] = bb_mid + BOLLINGER_STD * bb_std
    df["BB_lower"] = bb_mid - BOLLINGER_STD * bb_std
    df["BB_mid"] = bb_mid
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid
    df["BB_pct"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # --- Volume MA ---
    df["Volume_MA"] = volume.rolling(VOLUME_MA_PERIOD).mean()
    df["Volume_ratio"] = volume / df["Volume_MA"]

    # --- Daily returns ---
    df["Returns"] = close.pct_change()

    # --- Rolling volatility ---
    df["Volatility"] = df["Returns"].rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)

    # --- Momentum ---
    for period in MOMENTUM_PERIODS:
        df[f"Mom_{period}d"] = close.pct_change(period)

    # --- Trend strength (ADX-like: ratio of directional moves) ---
    df["Trend_up"] = (close > df["EMA_20"]).astype(int)
    df["Trend_strength"] = (close - df["EMA_50"]) / df["EMA_50"]

    # --- Beta vs benchmark ---
    if benchmark is not None:
        bench_returns = benchmark["Close"].pct_change().rename("bench_ret")
        merged = df["Returns"].to_frame("sym_ret").join(bench_returns, how="inner")
        rolling_cov = merged["sym_ret"].rolling(60).cov(merged["bench_ret"])
        rolling_var = merged["bench_ret"].rolling(60).var()
        df["Beta"] = rolling_cov / rolling_var
    else:
        df["Beta"] = np.nan

    return df.dropna(subset=["EMA_200"])  # drop warmup rows
