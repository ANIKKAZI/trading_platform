"""
Scenario Engine.
Generates probabilistic bull / base / bear forecast scenarios for a stock
over a configurable forward horizon (default: 5 months ≈ 110 trading days).

Inputs consumed:
    features        — feature-enriched OHLCV DataFrame (FeatureEngine output)
    strategy_output — result dict from StrategyManager.aggregate()
    ml_output       — float in [-1.0, +1.0] from MLPredictionEngine.predict()
    context         — dict from MarketContextEngine.build_context()
    sr_zones        — SRZones from compute_sr_zones() (optional)

Each scenario contains:
    probability             : float in (0, 1)
    expected_move           : formatted string, e.g. "+12%"
    expected_move_percent   : float, e.g. 12.0
    price_range             : [low, high] projected price band
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from core.sr_engine import SRZones

logger = logging.getLogger(__name__)

_DEFAULT_FORECAST_MONTHS: int = 5
_TRADING_DAYS_PER_MONTH: float = 21.0


class ScenarioEngine:
    """
    Generates three probabilistic scenarios (bull / base / bear) for a
    single stock using a blend of volatility-scaled drift, strategy bias,
    ML prediction, trend context, and S/R targets.
    """

    def generate_scenarios(
        self,
        features: pd.DataFrame,
        strategy_output: dict,
        ml_output: float,
        context: dict,
        sr_zones: "SRZones | None" = None,
        horizon_months: int = _DEFAULT_FORECAST_MONTHS,
    ) -> dict:
        """
        Compute bull / base / bear scenarios.

        Returns
        -------
        dict with keys: bull_case, base_case, bear_case.
        Each value is a dict:
            {probability, expected_move, expected_move_percent, price_range}
        """
        row = features.iloc[-1]
        current_price = float(row.get("Close", 0.0))
        if current_price <= 0:
            raise ValueError("current_price must be positive")

        ann_vol: float = float(context.get("volatility", 0.25))
        spy_trend: str = context.get("spy_trend", "neutral")
        trend_dir: str = context.get("trend_direction", "mixed")

        final_score: float = float(strategy_output.get("final_score", 0.0))
        triggered: list[str] = strategy_output.get("triggered", [])

        # --- Horizon-scaled volatility ---
        months = max(1, horizon_months)
        horizon_vol = (ann_vol / np.sqrt(12)) * np.sqrt(months)

        # --- Directional bias [-1, +1] ---
        strategy_bias = float(np.clip(final_score / 100.0, -1.0, 1.0))
        ml_bias = float(np.clip(ml_output, -1.0, 1.0))
        direction_bias = strategy_bias * 0.70 + ml_bias * 0.30

        # --- Trend multiplier (amplifies or dampens the bias) ---
        trend_mult = self._trend_multiplier(trend_dir, spy_trend)

        # --- Base drift: capped at ±40% ---
        base_drift = float(np.clip(direction_bias * horizon_vol * trend_mult, -0.40, 0.40))

        # --- Bull / bear spreads around base ---
        spread = horizon_vol * 1.5 * trend_mult
        bull_move = float(np.clip(base_drift + spread, 0.01, 0.90))
        bear_move = float(np.clip(base_drift - spread, -0.80, -0.01))

        # Nudge bull/bear toward nearest S/R targets when appropriate
        bull_target, bear_target = self._sr_targets(current_price, sr_zones)
        if bull_target is not None:
            sr_bull = (bull_target / current_price) - 1
            if sr_bull > 0:
                bull_move = max(bull_move, min(sr_bull, bull_move * 1.30))
        if bear_target is not None:
            sr_bear = (bear_target / current_price) - 1
            if sr_bear < 0:
                bear_move = min(bear_move, max(sr_bear, bear_move * 1.30))

        # --- Probabilities ---
        probs = self._compute_probabilities(direction_bias, trend_dir, spy_trend, triggered)

        # --- Price range half-width ---
        half_width = horizon_vol * 0.50 * current_price

        def _price_range(move: float) -> list[float]:
            mid = current_price * (1.0 + move)
            lo = max(0.01, round(mid - half_width, 2))
            hi = round(mid + half_width, 2)
            return [lo, hi]

        return {
            "bull_case": {
                "probability": probs["bull"],
                "expected_move": f"{bull_move:+.0%}",
                "expected_move_percent": round(bull_move * 100, 1),
                "price_range": _price_range(bull_move),
            },
            "base_case": {
                "probability": probs["base"],
                "expected_move": f"{base_drift:+.0%}",
                "expected_move_percent": round(base_drift * 100, 1),
                "price_range": _price_range(base_drift),
            },
            "bear_case": {
                "probability": probs["bear"],
                "expected_move": f"{bear_move:+.0%}",
                "expected_move_percent": round(bear_move * 100, 1),
                "price_range": _price_range(bear_move),
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trend_multiplier(self, trend_dir: str, spy_trend: str) -> float:
        """
        Returns a multiplier [0.50, 1.50] that scales the horizon_vol spread.
        Strong trends justify wider bull targets; strong downtrends widen bear targets
        while compressing the upside.
        """
        mult = 1.0
        trend_adjustments = {
            "strong_uptrend": +0.30,
            "uptrend": +0.15,
            "mixed": 0.0,
            "downtrend": -0.15,
            "strong_downtrend": -0.30,
        }
        mult += trend_adjustments.get(trend_dir, 0.0)
        if spy_trend == "bullish":
            mult += 0.10
        elif spy_trend == "bearish":
            mult -= 0.10
        return float(np.clip(mult, 0.50, 1.50))

    def _sr_targets(
        self,
        current_price: float,
        sr_zones: "SRZones | None",
    ) -> tuple[float | None, float | None]:
        """
        Extract nearest resistance price (for bull target) and nearest
        support price (for bear target) from SRZones.
        """
        bull_target: float | None = None
        bear_target: float | None = None
        if sr_zones is not None:
            if sr_zones.resistance:
                bull_target = float(sr_zones.resistance[0][0])
            if sr_zones.support:
                bear_target = float(sr_zones.support[0][1])
        return bull_target, bear_target

    def _compute_probabilities(
        self,
        direction_bias: float,
        trend_dir: str,
        spy_trend: str,
        triggered: list[str],
    ) -> dict[str, float]:
        """
        Compute bull / base / bear probabilities that sum to 1.0.
        Starting distribution: bull=0.25, base=0.55, bear=0.20.
        The direction_bias and trend context shift mass between the tails.
        """
        bull_p = 0.25 + direction_bias * 0.20
        bear_p = 0.20 - direction_bias * 0.15

        if trend_dir == "strong_uptrend":
            bull_p += 0.05
            bear_p -= 0.05
        elif trend_dir == "strong_downtrend":
            bull_p -= 0.05
            bear_p += 0.05

        if spy_trend == "bullish":
            bull_p += 0.03
        elif spy_trend == "bearish":
            bear_p += 0.03

        # Both momentum and breakout firing is a strong confirmation
        if "MomentumStrategy" in triggered and "BreakoutStrategy" in triggered:
            bull_p += 0.04

        # Hard clamps
        bull_p = float(np.clip(bull_p, 0.05, 0.65))
        bear_p = float(np.clip(bear_p, 0.05, 0.55))
        base_p = 1.0 - bull_p - bear_p

        # If base collapsed, rescale all three
        if base_p < 0.10:
            total = bull_p + bear_p + 0.10
            bull_p = round(bull_p / total, 2)
            bear_p = round(bear_p / total, 2)
            base_p = round(1.0 - bull_p - bear_p, 2)

        return {
            "bull": round(bull_p, 2),
            "base": round(max(0.10, base_p), 2),
            "bear": round(bear_p, 2),
        }
