"""
Breakout Strategy.
Detects resistance breakouts with volume confirmation.
"""

import pandas as pd
from strategies.base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "BreakoutStrategy"

    def generate_signal(self, data: pd.DataFrame, context: dict) -> int:
        row = data.iloc[-1]
        prev_row = data.iloc[-2] if len(data) >= 2 else None
        sr_zones = context.get("sr_zones")

        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = row["Volume_ratio"] > 1.5

        # Breakout above resistance zone
        if sr_zones and sr_zones.resistance:
            nearest_resistance_low, nearest_resistance_high = sr_zones.resistance[0]
            # Price closed above the resistance zone with volume
            if prev_row is not None:
                prev_close = prev_row["Close"]
                curr_close = row["Close"]
                broke_resistance = (
                    prev_close < nearest_resistance_high and
                    curr_close > nearest_resistance_high and
                    volume_confirmed
                )
                if broke_resistance:
                    return 1

        # Breakdown below support zone
        if sr_zones and sr_zones.support:
            nearest_support_low, nearest_support_high = sr_zones.support[0]
            if prev_row is not None:
                prev_close = prev_row["Close"]
                curr_close = row["Close"]
                broke_support = (
                    prev_close > nearest_support_low and
                    curr_close < nearest_support_low and
                    volume_confirmed
                )
                if broke_support:
                    return -1

        # Also check 52-week high breakout (no S/R needed)
        if len(data) >= 252:
            high_52w = data["High"].iloc[-252:].max()
            if row["Close"] >= high_52w * 0.99 and volume_confirmed:
                return 1

        return 0

    def confidence_score(self, data: pd.DataFrame, context: dict) -> float:
        row = data.iloc[-1]
        vol_ratio = row["Volume_ratio"]
        # Confidence scales with volume
        if vol_ratio > 3.0:
            return 0.90
        if vol_ratio > 2.0:
            return 0.75
        if vol_ratio > 1.5:
            return 0.60
        return 0.40
