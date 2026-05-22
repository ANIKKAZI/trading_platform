"""
Main orchestrator — runs the full pipeline end-to-end.

Execution order:
  1. Load universe
  2. Fetch historical + latest data
  3. Compute features
  4. Compute support/resistance zones
  5. Build context objects
  6. Run all active strategies
  7. Aggregate signals
  8. ML predictions (optional)
  9. Rank stocks
  10. Select TOP 10
  11. Generate insights
  12. Output results (JSON + CLI)
"""

import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config.settings import BENCHMARK_SYMBOL, TOP_N
from core.universe import load_universe
from core.data_engine import fetch_universe
from core.feature_engine import compute_features
from core.sr_engine import compute_sr_zones
from engine.strategy_manager import StrategyManager
from engine.ranking_engine import rank_stocks
from engine.ml_engine import MLPredictionEngine, predict_all, _is_ml_enabled
from engine.insight_engine import generate_all_insights
from engine.output_formatter import build_json_output, save_json_output
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.volatility_strategy import VolatilityStrategy

logger = logging.getLogger(__name__)


def _build_strategy_manager() -> StrategyManager:
    manager = StrategyManager()
    manager.register(MomentumStrategy())
    manager.register(MeanReversionStrategy())
    manager.register(BreakoutStrategy())
    manager.register(VolatilityStrategy())
    return manager


def _process_symbol(symbol: str, raw_data: dict, manager: StrategyManager) -> tuple[str, dict] | None:
    """Feature engineering + S/R + strategy aggregation for one symbol."""
    if symbol not in raw_data:
        return None
    try:
        bench_df = raw_data.get(BENCHMARK_SYMBOL)
        enriched = compute_features(raw_data[symbol], benchmark=bench_df)
        if enriched.empty or len(enriched) < 10:
            return None

        sr_zones = compute_sr_zones(enriched)
        context = {
            "symbol": symbol,
            "sr_zones": sr_zones,
        }
        agg = manager.aggregate(enriched, context)
        return symbol, {
            "data": enriched,
            "agg": agg,
            "sr_zones": sr_zones,
        }
    except Exception as exc:
        logger.warning("Failed to process %s: %s", symbol, exc)
        return None


def run(
    universe: str | None = None,
    force_refresh: bool = False,
    top_n: int = TOP_N,
    save_output: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Main entry point. Returns the JSON-serializable output dict.
    """
    # --- 1. Load universe ---
    symbols = load_universe(universe)
    logger.info("Universe: %d symbols", len(symbols))

    # --- 2. Fetch data ---
    logger.info("Fetching market data...")
    raw_data = fetch_universe(symbols, force_refresh=force_refresh)

    # Remove benchmark from symbol list if present
    analysis_symbols = [s for s in symbols if s in raw_data and s != BENCHMARK_SYMBOL]
    logger.info("%d symbols with data available", len(analysis_symbols))

    # --- 3-7. Feature engineering + strategies (parallel) ---
    manager = _build_strategy_manager()
    symbol_results: dict = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_process_symbol, sym, raw_data, manager): sym
            for sym in analysis_symbols
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                sym, payload = result
                symbol_results[sym] = payload

    logger.info("Processed %d symbols", len(symbol_results))

    # --- 8. ML predictions (optional) ---
    ml_predictions = None
    if _is_ml_enabled():
        ml_engine = MLPredictionEngine()
        enriched_map = {s: p["data"] for s, p in symbol_results.items()}
        ml_predictions = predict_all(ml_engine, enriched_map)

    # --- 9-10. Rank and select top N ---
    ranked = rank_stocks(symbol_results, ml_predictions=ml_predictions, top_n=top_n)
    logger.info("Top %d stocks ranked", len(ranked))

    # --- 11. Generate insights ---
    insights = generate_all_insights(ranked, ml_predictions)

    # --- 12. Build output ---
    output = build_json_output(ranked, ml_predictions)

    if verbose:
        print("\n" + "=" * 60)
        print(f"  QUANTITATIVE STOCK INTELLIGENCE — {output['date']}")
        print("=" * 60)
        for insight in insights:
            print(insight)
        print("\n--- JSON Summary ---")
        print(json.dumps(output, indent=2))

    if save_output:
        path = save_json_output(output)
        logger.info("Results saved to %s", path)

    return output
