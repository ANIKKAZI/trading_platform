"""
Strategy Manager — dynamic plugin registry and signal aggregator.

Usage:
    manager = StrategyManager()
    manager.register(MomentumStrategy())
    manager.register(BreakoutStrategy())
    result = manager.aggregate(data, context)
"""

import logging
from typing import Any
import pandas as pd

from strategies.base_strategy import BaseStrategy
from config.settings import STRATEGY_WEIGHTS

logger = logging.getLogger(__name__)


class StrategyManager:

    def __init__(self):
        self._strategies: dict[str, BaseStrategy] = {}
        self._enabled: dict[str, bool] = {}
        self._weights: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, strategy: BaseStrategy, weight: float | None = None) -> None:
        name = strategy.name
        self._strategies[name] = strategy
        self._enabled[name] = True
        self._weights[name] = weight if weight is not None else STRATEGY_WEIGHTS.get(name, 1.0)
        logger.info("Registered strategy: %s (weight=%.2f)", name, self._weights[name])

    def remove(self, name: str) -> None:
        self._strategies.pop(name, None)
        self._enabled.pop(name, None)
        self._weights.pop(name, None)
        logger.info("Removed strategy: %s", name)

    def enable(self, name: str) -> None:
        if name in self._enabled:
            self._enabled[name] = True

    def disable(self, name: str) -> None:
        if name in self._enabled:
            self._enabled[name] = False

    def list_strategies(self) -> list[dict]:
        return [
            {"name": n, "enabled": self._enabled[n], "weight": self._weights[n]}
            for n in self._strategies
        ]

    # ------------------------------------------------------------------
    # Signal aggregation
    # ------------------------------------------------------------------

    def aggregate(self, data: pd.DataFrame, context: dict) -> dict[str, Any]:
        """
        Run all enabled strategies and return aggregated result.

        Returns dict with:
            final_score     : weighted sum of signals × confidence
            raw_signals     : {strategy_name: signal}
            triggered       : list of strategy names that produced non-zero signals
            signal_label    : STRONG BUY | BUY | NEUTRAL | SELL | STRONG SELL
        """
        raw_signals: dict[str, int] = {}
        confidences: dict[str, float] = {}
        triggered: list[str] = []
        total_weight = 0.0
        weighted_sum = 0.0

        for name, strategy in self._strategies.items():
            if not self._enabled[name]:
                continue
            try:
                signal = strategy.generate_signal(data, context)
                conf = strategy.confidence_score(data, context)
            except Exception as exc:
                logger.warning("Strategy %s failed: %s", name, exc)
                signal, conf = 0, 0.5

            raw_signals[name] = signal
            confidences[name] = conf

            w = self._weights[name]
            weighted_sum += signal * conf * w
            total_weight += w

            if signal != 0:
                triggered.append(name)

        final_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 0.0

        return {
            "final_score": final_score,
            "raw_signals": raw_signals,
            "confidences": confidences,
            "triggered": triggered,
            "signal_label": _score_to_label(final_score),
        }


def _score_to_label(score: float) -> str:
    if score >= 60:
        return "STRONG BUY"
    if score >= 25:
        return "BUY"
    if score <= -60:
        return "STRONG SELL"
    if score <= -25:
        return "SELL"
    return "NEUTRAL"
