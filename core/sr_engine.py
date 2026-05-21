"""
Support and Resistance engine.
Detects swing highs/lows, clusters them into zones, and returns
support/resistance zone pairs as [zone_low, zone_high].
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from config.settings import SR_SWING_WINDOW, SR_CLUSTER_TOLERANCE, SR_MIN_TOUCHES


@dataclass
class SRZones:
    support: list[tuple[float, float]]
    resistance: list[tuple[float, float]]


def _find_swings(df: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of swing-high and swing-low price levels."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    swing_highs, swing_lows = [], []
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - window: i + window + 1]):
            swing_lows.append(lows[i])

    return np.array(swing_highs), np.array(swing_lows)


def _cluster_levels(levels: np.ndarray, tolerance: float, min_touches: int) -> list[tuple[float, float]]:
    """Cluster price levels within tolerance and return zones (low, high)."""
    if len(levels) == 0:
        return []

    sorted_levels = np.sort(levels)
    zones = []
    group = [sorted_levels[0]]

    for price in sorted_levels[1:]:
        if (price - group[0]) / group[0] <= tolerance:
            group.append(price)
        else:
            if len(group) >= min_touches:
                zones.append((float(np.min(group)), float(np.max(group))))
            group = [price]

    if len(group) >= min_touches:
        zones.append((float(np.min(group)), float(np.max(group))))

    return zones


def compute_sr_zones(df: pd.DataFrame, current_price: float | None = None) -> SRZones:
    """
    Compute support and resistance zones for a symbol's OHLCV DataFrame.
    If current_price is provided, support zones are below it and resistance above.
    """
    swing_highs, swing_lows = _find_swings(df, SR_SWING_WINDOW)

    all_levels = np.concatenate([swing_highs, swing_lows])
    all_zones = _cluster_levels(all_levels, SR_CLUSTER_TOLERANCE, SR_MIN_TOUCHES)

    if current_price is None:
        current_price = float(df["Close"].iloc[-1])

    support_zones = [z for z in all_zones if z[1] <= current_price]
    resistance_zones = [z for z in all_zones if z[0] >= current_price]

    # Sort: support descending (nearest first), resistance ascending (nearest first)
    support_zones = sorted(support_zones, key=lambda z: z[1], reverse=True)
    resistance_zones = sorted(resistance_zones, key=lambda z: z[0])

    return SRZones(support=support_zones, resistance=resistance_zones)
