"""
Mean Reversion Strategy.
Detects oversold/overbought conditions and fade-the-move setups.
"""

import pandas as pd
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "MeanReversionStrategy"

    def generate_signal(self, data: pd.DataFrame, context: dict) -> int:
        row = data.iloc[-1]

        rsi = row["RSI"]
        bb_pct = row["BB_pct"]
        price_vs_ema50 = (row["Close"] - row["EMA_50"]) / row["EMA_50"]

        # Oversold: buy signal
        oversold = (rsi < 35) and (bb_pct < 0.2) and (price_vs_ema50 < -0.05)
        # Overbought: sell signal
        overbought = (rsi > 65) and (bb_pct > 0.8) and (price_vs_ema50 > 0.05)

        # Mild oversold
        mild_oversold = (rsi < 45) and (bb_pct < 0.35)
        mild_overbought = (rsi > 55) and (bb_pct > 0.65)

        if oversold:
            return 1
        if overbought:
            return -1
        if mild_oversold:
            return 1
        if mild_overbought:
            return -1
        return 0

    def confidence_score(self, data: pd.DataFrame, context: dict) -> float:
        row = data.iloc[-1]
        rsi = row["RSI"]
        bb_pct = row["BB_pct"]

        # Higher confidence when more extreme readings
        if rsi < 30 or rsi > 70:
            return 0.85
        if rsi < 40 or rsi > 60:
            return 0.65
        if bb_pct < 0.2 or bb_pct > 0.8:
            return 0.60
        return 0.40
