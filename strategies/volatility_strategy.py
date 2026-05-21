"""
Volatility Strategy.
Signals based on ATR expansion/contraction and Bollinger Band width.
"""

import pandas as pd
from strategies.base_strategy import BaseStrategy


class VolatilityStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "VolatilityStrategy"

    def generate_signal(self, data: pd.DataFrame, context: dict) -> int:
        row = data.iloc[-1]

        # ATR expansion: volatility increasing (potential breakout setup)
        if len(data) >= 20:
            atr_sma = data["ATR"].iloc[-20:].mean()
            atr_expanding = row["ATR"] > atr_sma * 1.2
            atr_contracting = row["ATR"] < atr_sma * 0.8
        else:
            atr_expanding = False
            atr_contracting = False

        # BB squeeze: very narrow bands (coiling before move)
        if len(data) >= 50:
            bb_width_sma = data["BB_width"].iloc[-50:].mean()
            bb_squeeze = row["BB_width"] < bb_width_sma * 0.7
        else:
            bb_squeeze = False

        # Combine: ATR expanding + price above EMA20 = bullish vol expansion
        if atr_expanding and row["Close"] > row["EMA_20"]:
            return 1

        # ATR expanding + price below EMA20 = bearish vol expansion
        if atr_expanding and row["Close"] < row["EMA_20"]:
            return -1

        # BB squeeze + momentum alignment = anticipate breakout (mild bullish bias)
        if bb_squeeze and row["Close"] > row["EMA_50"]:
            return 1

        if bb_squeeze and row["Close"] < row["EMA_50"]:
            return -1

        return 0

    def confidence_score(self, data: pd.DataFrame, context: dict) -> float:
        row = data.iloc[-1]
        if len(data) >= 20:
            atr_sma = data["ATR"].iloc[-20:].mean()
            ratio = row["ATR"] / atr_sma if atr_sma > 0 else 1.0
            return min(0.4 + abs(ratio - 1.0) * 0.3, 0.9)
        return 0.5
