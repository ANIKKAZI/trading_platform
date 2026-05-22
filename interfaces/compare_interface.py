"""
Compare Interface — Stock Comparison & 5-Month Forecast.

Runs the full existing pipeline for two stocks in parallel, then passes
their outputs through the new forecast and comparison engines to produce
a side-by-side projection and a winner declaration.

CLI usage
---------
    python -m interfaces.compare_interface NVDA AAPL
    python -m interfaces.compare_interface NVDA AAPL --horizon 3 --refresh

Programmatic usage
------------------
    from interfaces.compare_interface import CompareInterface
    interface = CompareInterface()
    result = interface.run("NVDA", "AAPL")
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config.settings import BENCHMARK_SYMBOL
from core.data_engine import fetch_symbol
from core.feature_engine import compute_features
from core.sr_engine import compute_sr_zones
from engine.comparison_engine import ComparisonEngine
from engine.forecast_engine import ForecastEngine
from engine.insight_engine import generate_comparison_insight
from engine.ml_engine import MLPredictionEngine
from engine.output_formatter import save_comparison_output
from engine.strategy_manager import StrategyManager
from strategies.breakout_strategy import BreakoutStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.volatility_strategy import VolatilityStrategy

logger = logging.getLogger(__name__)

_DEFAULT_HORIZON: int = 5
_BAR = "=" * 60
_SEP = "-" * 50


# ---------------------------------------------------------------------------
# Module-level factory (mirrors orchestrator.py convention)
# ---------------------------------------------------------------------------

def _sr_zones_to_dict(sr_zones) -> dict:
    """Convert SRZones to a plain serialisable dict (safe for st.cache_data)."""
    if sr_zones is None:
        return {"support": [], "resistance": []}
    return {
        "support": [list(z) for z in (sr_zones.support or [])],
        "resistance": [list(z) for z in (sr_zones.resistance or [])],
    }


def _build_strategy_manager() -> StrategyManager:
    manager = StrategyManager()
    manager.register(MomentumStrategy())
    manager.register(MeanReversionStrategy())
    manager.register(BreakoutStrategy())
    manager.register(VolatilityStrategy())
    return manager


# ---------------------------------------------------------------------------
# Main interface class
# ---------------------------------------------------------------------------

class CompareInterface:
    """
    Entry point for the stock comparison + 5-month forecast workflow.

    Reuses every existing engine unchanged and layers the new forecast /
    comparison engines on top.
    """

    def __init__(self, force_refresh: bool = False) -> None:
        self._force_refresh = force_refresh
        self._strategy_manager = _build_strategy_manager()
        self._ml_engine = MLPredictionEngine()
        self._forecast_engine = ForecastEngine()
        self._comparison_engine = ComparisonEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbol_a: str,
        symbol_b: str,
        horizon_months: int = _DEFAULT_HORIZON,
        print_output: bool = True,
        save_output: bool = True,
    ) -> dict:
        """
        Execute the full comparison pipeline for two symbols.

        Parameters
        ----------
        symbol_a / symbol_b : ticker strings (case-insensitive)
        horizon_months      : forecast horizon in months (default 5)
        print_output        : write formatted display to stdout
        save_output         : persist JSON to output/ directory

        Returns
        -------
        Full structured result dict.
        """
        symbol_a = symbol_a.upper()
        symbol_b = symbol_b.upper()
        logger.info(
            "Comparison pipeline started: %s vs %s  (%d-month horizon)",
            symbol_a, symbol_b, horizon_months,
        )

        # Step 1 — fetch benchmark once; fetch both stocks in parallel
        benchmark_data = self._fetch_safe(BENCHMARK_SYMBOL)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(self._build_payload, symbol_a, benchmark_data)
            fut_b = pool.submit(self._build_payload, symbol_b, benchmark_data)
            payload_a = fut_a.result()
            payload_b = fut_b.result()

        # Step 2 — generate 5-month forecast for each stock
        forecast_a = self._forecast_engine.forecast(
            symbol=symbol_a,
            features=payload_a["features"],
            strategy_output=payload_a["strategy_output"],
            ml_output=payload_a["ml_output"],
            sr_zones=payload_a["sr_zones"],
            benchmark_data=benchmark_data,
            horizon_months=horizon_months,
        )
        forecast_b = self._forecast_engine.forecast(
            symbol=symbol_b,
            features=payload_b["features"],
            strategy_output=payload_b["strategy_output"],
            ml_output=payload_b["ml_output"],
            sr_zones=payload_b["sr_zones"],
            benchmark_data=benchmark_data,
            horizon_months=horizon_months,
        )

        # Step 3 — merge payloads with forecast data for comparison engine
        comp_a = self._merge_payload(symbol_a, payload_a, forecast_a)
        comp_b = self._merge_payload(symbol_b, payload_b, forecast_b)

        # Step 4 — compare
        comparison = self._comparison_engine.compare(comp_a, comp_b)

        # Step 5 — generate insights
        insight_a = generate_comparison_insight(symbol_a, comp_a, forecast_a, comparison)
        insight_b = generate_comparison_insight(symbol_b, comp_b, forecast_b, comparison)

        # Step 6 — assemble final result
        result = self._assemble_result(
            symbol_a, symbol_b,
            forecast_a, forecast_b,
            comparison,
            insight_a, insight_b,
            horizon_months,
            comp_a, comp_b,
        )

        if save_output:
            path = save_comparison_output(result)
            logger.info("Comparison saved → %s", path)

        if print_output:
            self._print_display(result, forecast_a, forecast_b, payload_a, payload_b)

        return result

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _fetch_safe(self, symbol: str) -> pd.DataFrame | None:
        try:
            return fetch_symbol(symbol, force_refresh=self._force_refresh)
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", symbol, exc)
            return None

    def _build_payload(
        self,
        symbol: str,
        benchmark_data: pd.DataFrame | None,
    ) -> dict:
        """
        Run the existing pipeline stages (data → features → S/R → strategies → ML)
        for a single symbol and return a unified payload dict.
        """
        raw = self._fetch_safe(symbol)
        if raw is None or raw.empty:
            raise ValueError(f"No data available for {symbol}")

        features = compute_features(raw, benchmark=benchmark_data)

        current_price = float(features.iloc[-1]["Close"])
        sr_zones = compute_sr_zones(features, current_price=current_price)

        strategy_context: dict = {
            "symbol": symbol,
            "sr_zones": sr_zones,
            "current_price": current_price,
        }
        strategy_output = self._strategy_manager.aggregate(features, strategy_context)
        ml_output = self._ml_engine.predict(features)

        return {
            "symbol": symbol,
            "raw": raw,
            "features": features,
            "sr_zones": sr_zones,
            "strategy_output": strategy_output,
            "ml_output": ml_output,
            "latest_row": features.iloc[-1],
        }

    @staticmethod
    def _merge_payload(symbol: str, payload: dict, forecast: dict) -> dict:
        """
        Merge pipeline payload with forecast output so that the comparison
        engine and insight engine have a single unified dict to work from.
        """
        return {
            **payload,
            "symbol": symbol,
            "scenarios": forecast["scenarios"],
            "market_context": forecast["market_context"],
            "risk_factors": forecast["risk_factors"],
            "trend_summary": forecast["trend_summary"],
        }

    @staticmethod
    def _assemble_result(
        sym_a: str,
        sym_b: str,
        forecast_a: dict,
        forecast_b: dict,
        comparison: dict,
        insight_a: str,
        insight_b: str,
        horizon_months: int,
        comp_a: dict | None = None,
        comp_b: dict | None = None,
    ) -> dict:
        def _short_forecast(sym: str, fc: dict) -> dict:
            sc = fc["scenarios"]
            return {
                "bull": sc["bull_case"]["expected_move"],
                "base": sc["base_case"]["expected_move"],
                "bear": sc["bear_case"]["expected_move"],
            }

        def _extract_signals(comp: dict | None) -> dict:
            if comp is None:
                return {}
            so = comp.get("strategy_output", {})
            return {
                "raw_signals": so.get("raw_signals", {}),
                "confidences": so.get("confidences", {}),
                "triggered": so.get("triggered", []),
                "final_score": float(so.get("final_score", 0.0)),
                "signal_label": so.get("signal_label", "NEUTRAL"),
            }

        return {
            "date": datetime.date.today().isoformat(),
            "stock_a": sym_a,
            "stock_b": sym_b,
            "horizon_months": horizon_months,
            "forecast": {
                sym_a: _short_forecast(sym_a, forecast_a),
                sym_b: _short_forecast(sym_b, forecast_b),
            },
            "detailed_forecast": {
                sym_a: forecast_a,
                sym_b: forecast_b,
            },
            "winner": comparison["winner"],
            "confidence": f"{comparison['confidence']}%",
            "key_factors": comparison["key_factors"],
            "comparison_summary": comparison["comparison_summary"],
            "insights": {
                sym_a: insight_a,
                sym_b: insight_b,
            },
            # Additional data consumed by the visual dashboard
            "strategy_signals": {
                sym_a: _extract_signals(comp_a),
                sym_b: _extract_signals(comp_b),
            },
            "ml_outputs": {
                sym_a: float(comp_a.get("ml_output", 0.0)) if comp_a else 0.0,
                sym_b: float(comp_b.get("ml_output", 0.0)) if comp_b else 0.0,
            },
            "sr_zone_data": {
                sym_a: _sr_zones_to_dict(comp_a.get("sr_zones") if comp_a else None),
                sym_b: _sr_zones_to_dict(comp_b.get("sr_zones") if comp_b else None),
            },
        }

    # ------------------------------------------------------------------
    # CLI display
    # ------------------------------------------------------------------

    def _print_display(
        self,
        result: dict,
        forecast_a: dict,
        forecast_b: dict,
        payload_a: dict,
        payload_b: dict,
    ) -> None:
        sym_a = result["stock_a"]
        sym_b = result["stock_b"]
        winner = result["winner"]
        confidence = result["confidence"]
        horizon = result["horizon_months"]

        print(f"\n{_BAR}")
        print("  STOCK COMPARISON FORECAST")
        print(f"  {sym_a}  vs  {sym_b}  |  {horizon}-Month Projection")
        print(_BAR)

        for sym, fc, payload in [
            (sym_a, forecast_a, payload_a),
            (sym_b, forecast_b, payload_b),
        ]:
            sc = fc["scenarios"]
            sr = payload.get("sr_zones")
            print(f"\n  {sym}")
            print(f"    Current Price : ${fc['current_price']:.2f}")
            print(
                f"    Bull Case     : {sc['bull_case']['expected_move']:<8} "
                f"  (p={sc['bull_case']['probability']:.0%})"
            )
            print(
                f"    Base Case     : {sc['base_case']['expected_move']:<8} "
                f"  (p={sc['base_case']['probability']:.0%})"
            )
            print(
                f"    Bear Case     : {sc['bear_case']['expected_move']:<8} "
                f"  (p={sc['bear_case']['probability']:.0%})"
            )

            if sr is not None:
                if sr.support:
                    lo, hi = sr.support[0]
                    print(f"    Support       : {lo:.2f}–{hi:.2f}")
                if sr.resistance:
                    lo, hi = sr.resistance[0]
                    print(f"    Resistance    : {lo:.2f}–{hi:.2f}")

            risks = fc.get("risk_factors", [])
            if risks and risks[0] != "No significant risk flags identified":
                print(f"    ⚠ Risk        : {risks[0]}")

            print(f"\n  {_SEP}")

        print(f"\n  Winner     : {winner}")
        print(f"  Confidence : {confidence}")
        print(f"\n  Key Factors:")
        for factor in result.get("key_factors", []):
            print(f"    ✓ {factor}")

        print(f"\n{_BAR}")
        print("  DETAILED INSIGHTS")
        print(_BAR)
        print(result["insights"][sym_a])
        print()
        print(result["insights"][sym_b])
        print(f"\n{_BAR}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock Comparison & 5-Month Probabilistic Forecast"
    )
    parser.add_argument("stock_a", type=str.upper, help="First ticker symbol (e.g. NVDA)")
    parser.add_argument("stock_b", type=str.upper, help="Second ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--horizon", type=int, default=_DEFAULT_HORIZON,
        help=f"Forecast horizon in months (default: {_DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force fresh data download (ignore cache)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip writing the JSON output file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    interface = CompareInterface(force_refresh=args.refresh)
    interface.run(
        symbol_a=args.stock_a,
        symbol_b=args.stock_b,
        horizon_months=args.horizon,
        print_output=True,
        save_output=not args.no_save,
    )


if __name__ == "__main__":
    main()
