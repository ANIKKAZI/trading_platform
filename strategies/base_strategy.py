"""
Base class for all trading strategies.
Every strategy must subclass BaseStrategy and implement generate_signal().
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    Interface every strategy plugin must implement.

    generate_signal() returns:
        +1  = bullish
         0  = neutral / no signal
        -1  = bearish
    confidence_score() returns a float in [0.0, 1.0] (optional, defaults to 0.5).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, context: dict) -> int:
        """
        data    : feature-enriched OHLCV DataFrame for one symbol (tail used)
        context : dict with keys like 'sr_zones', 'symbol', 'benchmark_data'
        Returns : -1 | 0 | +1
        """
        ...

    def confidence_score(self, data: pd.DataFrame, context: dict) -> float:
        """Override to provide strategy confidence. Default: 0.5."""
        return 0.5
