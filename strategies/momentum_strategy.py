"""
Momentum Strategy.
Trend-following logic using EMA alignment and multi-period momentum.
"""

import pandas as pd
from strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "MomentumStrategy"

    def generate_signal(self, data: pd.DataFrame, context: dict) -> int:
        row = data.iloc[-1]

        # EMA alignment: price > EMA20 > EMA50 > EMA200
        ema_aligned_bull = (
            row["Close"] > row["EMA_20"] > row["EMA_50"] > row["EMA_200"]
        )
        ema_aligned_bear = (
            row["Close"] < row["EMA_20"] < row["EMA_50"] < row["EMA_200"]
        )

        # Momentum across multiple timeframes
        mom_bull = (row["Mom_5d"] > 0) and (row["Mom_20d"] > 0) and (row["Mom_60d"] > 0)
        mom_bear = (row["Mom_5d"] < 0) and (row["Mom_20d"] < 0) and (row["Mom_60d"] < 0)

        if ema_aligned_bull and mom_bull:
            return 1
        if ema_aligned_bear and mom_bear:
            return -1
        # Partial bullish
        if ema_aligned_bull or (mom_bull and row["Close"] > row["EMA_50"]):
            return 1
        if ema_aligned_bear or (mom_bear and row["Close"] < row["EMA_50"]):
            return -1
        return 0

    def confidence_score(self, data: pd.DataFrame, context: dict) -> float:
        row = data.iloc[-1]
        score = 0.5
        if row["Close"] > row["EMA_20"]:
            score += 0.1
        if row["EMA_20"] > row["EMA_50"]:
            score += 0.1
        if row["EMA_50"] > row["EMA_200"]:
            score += 0.1
        if row["Mom_5d"] > 0 and row["Mom_20d"] > 0:
            score += 0.1
        if row["Mom_60d"] > 0:
            score += 0.1
        return min(score, 1.0)
