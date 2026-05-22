"""
Output serializer.
Converts ranked stock results to the mandatory JSON output format.
"""

import json
import os
import datetime

from config.settings import OUTPUT_DIR


def _zone_to_list(zone_tuple):
    if zone_tuple is None:
        return []
    return [round(zone_tuple[0], 2), round(zone_tuple[1], 2)]


def build_json_output(ranked_stocks: list[dict], ml_predictions: dict | None = None) -> dict:
    today = datetime.date.today().isoformat()
    top_10 = []

    for entry in ranked_stocks:
        sr = entry.get("sr_zones")
        support = _zone_to_list(sr.support[0] if (sr and sr.support) else None)
        resistance = _zone_to_list(sr.resistance[0] if (sr and sr.resistance) else None)

        ml_score = None
        if ml_predictions and entry["symbol"] in ml_predictions:
            raw = ml_predictions[entry["symbol"]]
            ml_score = f"{raw * 3:+.1f}%"  # scale to ≈±3%

        top_10.append({
            "symbol": entry["symbol"],
            "score": entry["score"],
            "signal": entry["signal"],
            "support": support,
            "resistance": resistance,
            "strategies_triggered": entry["triggered"],
            "prediction": ml_score,
            "strategy_score": entry["strategy_score"],
            "momentum_score": entry["momentum_score"],
        })

    return {"date": today, "top_10_stocks": top_10}


def save_json_output(output: dict, path: str | None = None) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(OUTPUT_DIR, f"results_{output['date']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# Comparison result serialiser (added for compare_interface)
# ---------------------------------------------------------------------------

def build_comparison_json(result: dict) -> dict:
    """
    Slim-down the full comparison result into the mandatory JSON output shape.

    Input:  the dict returned by CompareInterface._build_result()
    Output: serialisable dict suitable for writing to disk.
    """
    sym_a: str = result["stock_a"]
    sym_b: str = result["stock_b"]

    def _scenarios(sym: str) -> dict:
        fc = result.get("detailed_forecast", {}).get(sym, {})
        sc = fc.get("scenarios", {})
        return {
            "bull": sc.get("bull_case", {}).get("expected_move", "N/A"),
            "base": sc.get("base_case", {}).get("expected_move", "N/A"),
            "bear": sc.get("bear_case", {}).get("expected_move", "N/A"),
            "bull_probability": sc.get("bull_case", {}).get("probability", 0.0),
            "base_probability": sc.get("base_case", {}).get("probability", 0.0),
            "bear_probability": sc.get("bear_case", {}).get("probability", 0.0),
            "bull_price_range": sc.get("bull_case", {}).get("price_range", []),
            "base_price_range": sc.get("base_case", {}).get("price_range", []),
            "bear_price_range": sc.get("bear_case", {}).get("price_range", []),
            "current_price": fc.get("current_price"),
            "trend_summary": fc.get("trend_summary", ""),
            "risk_factors": fc.get("risk_factors", []),
        }

    return {
        "date": result.get("date", datetime.date.today().isoformat()),
        "stock_a": sym_a,
        "stock_b": sym_b,
        "horizon_months": result.get("horizon_months", 5),
        "forecast": {
            sym_a: _scenarios(sym_a),
            sym_b: _scenarios(sym_b),
        },
        "winner": result.get("winner", ""),
        "confidence": result.get("confidence", "N/A"),
        "key_factors": result.get("key_factors", []),
        "comparison_summary": result.get("comparison_summary", []),
    }


def save_comparison_output(result: dict, path: str | None = None) -> str:
    """
    Serialise and write a comparison result to
    ``output/comparison_results_YYYY-MM-DD.json``.

    Returns the path of the written file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = result.get("date", datetime.date.today().isoformat())
    if path is None:
        path = os.path.join(OUTPUT_DIR, f"comparison_results_{date_str}.json")
    payload = build_comparison_json(result)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
