"""
Market Context Engine.
Builds macro and market context for a symbol using SPY benchmark data
and EMA-based trend analysis.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Rough sector mapping for common symbols; unknown symbols fall back to "Unknown"
_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "GS": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "SPY": "Broad Market",
    "QQQ": "Technology ETF",
}


class MarketContextEngine:
    """
    Builds a market context dictionary for a single symbol.

    Context includes:
        symbol              : ticker
        beta                : precomputed beta from feature engine
        volatility          : annualized volatility
        sector              : mapped sector string
        spy_correlation     : Pearson correlation of daily returns vs SPY
        spy_trend           : 'bullish' | 'bearish' | 'neutral'
        trend_direction     : EMA-alignment-based trend label
    """

    def build_context(
        self,
        symbol: str,
        features: pd.DataFrame,
        benchmark_data: pd.DataFrame | None = None,
    ) -> dict:
        """
        Build market context for a symbol.

        Parameters
        ----------
        symbol          : ticker string
        features        : feature-enriched OHLCV DataFrame (output of feature engine)
        benchmark_data  : raw SPY OHLCV DataFrame (optional — falls back to defaults)

        Returns
        -------
        dict with keys: symbol, beta, volatility, sector, spy_correlation,
                        spy_trend, trend_direction
        """
        row = features.iloc[-1]

        context: dict = {
            "symbol": symbol,
            "beta": float(row.get("Beta", 1.0)),
            "volatility": float(row.get("Volatility", 0.25)),
            "sector": _SECTOR_MAP.get(symbol, "Unknown"),
        }

        if benchmark_data is not None and not benchmark_data.empty:
            context["spy_correlation"] = self._compute_correlation(features, benchmark_data)
            context["spy_trend"] = self._spy_trend(benchmark_data)
        else:
            context["spy_correlation"] = 0.70
            context["spy_trend"] = "neutral"

        context["trend_direction"] = self._classify_trend(row)
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_trend(self, row: pd.Series) -> str:
        """Classify price trend using EMA alignment."""
        close = float(row.get("Close", 0))
        ema20 = float(row.get("EMA_20", 0))
        ema50 = float(row.get("EMA_50", 0))
        ema200 = float(row.get("EMA_200", 0))

        if close > ema20 > ema50 > ema200:
            return "strong_uptrend"
        if close > ema50 > ema200:
            return "uptrend"
        if close < ema20 < ema50 < ema200:
            return "strong_downtrend"
        if close < ema50 < ema200:
            return "downtrend"
        return "mixed"

    def _compute_correlation(
        self, features: pd.DataFrame, benchmark: pd.DataFrame
    ) -> float:
        """
        Pearson correlation of daily returns vs SPY over the last 252 trading days.
        Falls back to 0.70 on any failure.
        """
        try:
            sym_ret = features["Close"].pct_change().dropna()
            spy_ret = benchmark["Close"].pct_change().dropna()
            aligned = pd.concat([sym_ret, spy_ret], axis=1, join="inner").dropna()
            if len(aligned) < 30:
                return 0.70
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            return round(corr, 3)
        except Exception as exc:
            logger.warning("Correlation computation failed for symbol: %s", exc)
            return 0.70

    def _spy_trend(self, benchmark: pd.DataFrame) -> str:
        """Determine SPY trend using 20-day / 50-day EMA cross."""
        if len(benchmark) < 50:
            return "neutral"
        close = benchmark["Close"]
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        current = float(close.iloc[-1])

        if current > ema20 > ema50:
            return "bullish"
        if current < ema20 < ema50:
            return "bearish"
        return "neutral"
