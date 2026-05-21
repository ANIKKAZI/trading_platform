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
