"""
Forecast Engine.
Orchestrates 5-month probabilistic forecasting for a single symbol.
Wraps ScenarioEngine and MarketContextEngine, then attaches risk factors
and a human-readable trend summary to the result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from engine.market_context_engine import MarketContextEngine
from engine.scenario_engine import ScenarioEngine

if TYPE_CHECKING:
    from core.sr_engine import SRZones

logger = logging.getLogger(__name__)

_DEFAULT_HORIZON_MONTHS: int = 5

_TREND_LABELS: dict[str, str] = {
    "strong_uptrend": "Strong uptrend",
    "uptrend": "Uptrend",
    "mixed": "Mixed / consolidating",
    "downtrend": "Downtrend",
    "strong_downtrend": "Strong downtrend",
}


class ForecastEngine:
    """
    Produces a full forecast package for a single symbol.

    Returned dict shape
    -------------------
    {
        symbol          : str,
        current_price   : float,
        horizon_months  : int,
        market_context  : dict,          # from MarketContextEngine
        scenarios       : dict,          # from ScenarioEngine (bull/base/bear)
        risk_factors    : list[str],
        trend_summary   : str,
    }
    """

    def __init__(self) -> None:
        self._context_engine = MarketContextEngine()
        self._scenario_engine = ScenarioEngine()

    def forecast(
        self,
        symbol: str,
        features: pd.DataFrame,
        strategy_output: dict,
        ml_output: float,
        sr_zones: "SRZones | None" = None,
        benchmark_data: pd.DataFrame | None = None,
        horizon_months: int = _DEFAULT_HORIZON_MONTHS,
    ) -> dict:
        """
        Run the full forecasting sequence for one symbol.

        Parameters
        ----------
        symbol          : ticker string
        features        : enriched OHLCV DataFrame (FeatureEngine output)
        strategy_output : result dict from StrategyManager.aggregate()
        ml_output       : float in [-1.0, +1.0] from MLPredictionEngine.predict()
        sr_zones        : SRZones object (optional)
        benchmark_data  : raw SPY OHLCV DataFrame (optional)
        horizon_months  : forecast horizon in months (default 5)

        Returns
        -------
        Full forecast dict (see class docstring).
        """
        context = self._context_engine.build_context(
            symbol=symbol,
            features=features,
            benchmark_data=benchmark_data,
        )

        scenarios = self._scenario_engine.generate_scenarios(
            features=features,
            strategy_output=strategy_output,
            ml_output=ml_output,
            context=context,
            sr_zones=sr_zones,
            horizon_months=horizon_months,
        )

        row = features.iloc[-1]
        current_price = float(row.get("Close", 0.0))
        risk_factors = self._identify_risk_factors(row, context, sr_zones)
        trend_summary = self._build_trend_summary(context, strategy_output)

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "horizon_months": horizon_months,
            "market_context": context,
            "scenarios": scenarios,
            "risk_factors": risk_factors,
            "trend_summary": trend_summary,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _identify_risk_factors(
        self,
        row: pd.Series,
        context: dict,
        sr_zones: "SRZones | None",
    ) -> list[str]:
        """
        Identify notable risk flags for the symbol. Returns a non-empty list
        (falls back to 'No significant risk flags identified' when clean).
        """
        risks: list[str] = []

        rsi = float(row.get("RSI", 50.0))
        vol = float(context.get("volatility", 0.20))
        beta = float(context.get("beta", 1.0))
        current = float(row.get("Close", 0.0))

        if rsi > 70:
            risks.append(f"RSI overbought ({rsi:.0f}) — elevated pullback risk")
        if rsi < 30:
            risks.append(f"RSI oversold ({rsi:.0f}) — may extend lower before recovery")

        if vol > 0.50:
            risks.append(f"High annualized volatility ({vol:.0%}) — wider expected price ranges")
        elif vol > 0.35:
            risks.append(f"Elevated volatility ({vol:.0%}) — increased forecast uncertainty")

        if abs(beta) > 1.5:
            risks.append(
                f"High beta ({beta:.2f}) — amplified sensitivity to broad-market moves"
            )

        if sr_zones is not None and sr_zones.resistance and current > 0:
            nearest_res = float(sr_zones.resistance[0][0])
            proximity = (nearest_res - current) / current
            if proximity < 0.03:
                risks.append(
                    f"Price within 3% of key resistance ({nearest_res:.2f}) — "
                    "breakout confirmation or rejection expected"
                )

        if context.get("spy_trend") == "bearish":
            risks.append("SPY in downtrend — broad-market headwind present")

        bb_pct = float(row.get("BB_pct", 0.5))
        if bb_pct > 0.90:
            risks.append("Price near upper Bollinger Band — overextension risk")
        elif bb_pct < 0.10:
            risks.append("Price near lower Bollinger Band — potential downside continuation")

        if not risks:
            risks.append("No significant risk flags identified")

        return risks

    def _build_trend_summary(self, context: dict, strategy_output: dict) -> str:
        """Single-line trend summary combining EMA trend, SPY trend, and strategy label."""
        td = _TREND_LABELS.get(context.get("trend_direction", "mixed"), "Mixed")
        spy = context.get("spy_trend", "neutral").capitalize()
        label = strategy_output.get("signal_label", "NEUTRAL")
        return f"{td}  |  SPY: {spy}  |  Signal: {label}"
