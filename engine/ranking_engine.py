"""
Stock Ranking Engine.
Scores all symbols in the universe and returns the top-N ranked stocks.
"""

import logging
import pandas as pd

from config.settings import TOP_N, ML_ENABLED

logger = logging.getLogger(__name__)


def _momentum_score(row: pd.Series) -> float:
    """Normalized momentum contribution [0, 100]."""
    score = 0.0
    if row.get("Mom_5d", 0) > 0:
        score += 20
    if row.get("Mom_20d", 0) > 0:
        score += 30
    if row.get("Mom_60d", 0) > 0:
        score += 30
    if row.get("Close", 0) > row.get("EMA_50", float("inf")):
        score += 20
    return score


def _volatility_adjustment(row: pd.Series) -> float:
    """
    Penalize extremely high volatility, reward moderate volatility.
    Returns adjustment in [-20, +10].
    """
    vol = row.get("Volatility", 0.2)
    if vol < 0.15:
        return 5     # low vol, slight bonus (stable trend)
    if vol < 0.30:
        return 10    # sweet spot
    if vol < 0.50:
        return 0
    return -15       # very high vol penalty


def _breakout_probability(row: pd.Series, sr_zones) -> float:
    """Simple proximity-to-resistance breakout score [0, 20]."""
    if sr_zones is None or not sr_zones.resistance:
        return 0.0
    current_price = row.get("Close", 0)
    nearest_res_low = sr_zones.resistance[0][0]
    if nearest_res_low <= 0:
        return 0.0
    proximity = (nearest_res_low - current_price) / nearest_res_low
    if proximity < 0.01:
        return 20.0    # very close to resistance
    if proximity < 0.03:
        return 12.0
    if proximity < 0.05:
        return 6.0
    return 0.0


def rank_stocks(
    symbol_results: dict,
    ml_predictions: dict | None = None,
    top_n: int = TOP_N,
) -> list[dict]:
    """
    symbol_results: {symbol: {"data": df, "agg": agg_result, "sr_zones": zones}}
    ml_predictions: {symbol: float} — optional ML score contribution

    Returns sorted list of dicts (descending score), top_n items.
    """
    scores = []

    for symbol, payload in symbol_results.items():
        agg = payload["agg"]
        data = payload["data"]
        sr_zones = payload.get("sr_zones")
        row = data.iloc[-1]

        # Base: strategy aggregate score maps to [0, 100] via normalization
        strategy_contribution = (agg["final_score"] + 100) / 2  # shift [-100,100] → [0,100]

        mom = _momentum_score(row)
        vol_adj = _volatility_adjustment(row)
        bp = _breakout_probability(row, sr_zones)

        ml_score = 0.0
        if ML_ENABLED and ml_predictions and symbol in ml_predictions:
            ml_score = float(ml_predictions[symbol]) * 20  # scale to [-20, +20]

        # Weighted composite score
        composite = (
            strategy_contribution * 0.45
            + mom * 0.25
            + bp * 0.15
            + vol_adj * 0.10
            + ml_score * 0.05
        )

        scores.append({
            "symbol": symbol,
            "score": round(composite, 2),
            "signal": agg["signal_label"],
            "strategy_score": round(agg["final_score"], 2),
            "momentum_score": round(mom, 2),
            "volatility_adj": round(vol_adj, 2),
            "breakout_prob": round(bp, 2),
            "ml_score": round(ml_score, 2),
            "triggered": agg["triggered"],
            "raw_signals": agg["raw_signals"],
            "confidences": agg["confidences"],
            "sr_zones": sr_zones,
            "latest_row": row,
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]
