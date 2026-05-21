"""
Insight Generation Engine.
Produces human-readable explanations for each top-ranked stock.
"""

import pandas as pd
from core.sr_engine import SRZones


def _format_zone(zone: tuple[float, float]) -> str:
    return f"[{zone[0]:.2f}, {zone[1]:.2f}]"


def generate_insight(
    symbol: str,
    ranked_entry: dict,
    ml_score: float | None = None,
) -> str:
    row: pd.Series = ranked_entry["latest_row"]
    triggered: list[str] = ranked_entry.get("triggered", [])
    raw_signals: dict = ranked_entry.get("raw_signals", {})
    confidences: dict = ranked_entry.get("confidences", {})
    sr_zones: SRZones | None = ranked_entry.get("sr_zones")
    score = ranked_entry["score"]
    signal_label = ranked_entry["signal"]

    lines = [f"{'='*60}", f"  {symbol}  |  Score: {score:.1f}  |  Signal: {signal_label}", f"{'='*60}"]

    # Strategy signals
    lines.append("Strategy Signals:")
    for name, sig in raw_signals.items():
        direction = "Bullish" if sig == 1 else ("Bearish" if sig == -1 else "Neutral")
        conf = confidences.get(name, 0.5)
        lines.append(f"  {name:<28} {direction:<10}  (conf: {conf:.0%})")

    # Key technical levels
    lines.append("\nKey Technicals:")
    lines.append(f"  Close:  {row['Close']:.2f}")
    lines.append(f"  RSI:    {row.get('RSI', float('nan')):.1f}")
    lines.append(f"  EMA20:  {row.get('EMA_20', float('nan')):.2f}  |  EMA50: {row.get('EMA_50', float('nan')):.2f}  |  EMA200: {row.get('EMA_200', float('nan')):.2f}")
    lines.append(f"  MACD hist: {row.get('MACD_hist', float('nan')):.3f}")
    lines.append(f"  ATR%:   {row.get('ATR_pct', float('nan')):.2%}")
    lines.append(f"  Vol ratio: {row.get('Volume_ratio', float('nan')):.2f}x")
    lines.append(f"  5D Mom: {row.get('Mom_5d', float('nan')):.2%}  |  20D: {row.get('Mom_20d', float('nan')):.2%}  |  60D: {row.get('Mom_60d', float('nan')):.2%}")

    # Support & Resistance zones
    if sr_zones:
        if sr_zones.resistance:
            lines.append(f"\n  Nearest Resistance: {_format_zone(sr_zones.resistance[0])}")
        if sr_zones.support:
            lines.append(f"  Nearest Support:    {_format_zone(sr_zones.support[0])}")

    # ML prediction
    if ml_score is not None:
        pct_pred = ml_score * 3  # rough ±3% scale
        direction = "positive" if pct_pred > 0 else "negative"
        lines.append(f"\n  Prediction bias: {pct_pred:+.1f}% expected ({direction})")

    # Risk factors
    lines.append("\nRisk Factors:")
    rsi = row.get("RSI", 50)
    vol = row.get("Volatility", 0)
    beta = row.get("Beta", 1)
    if rsi > 70:
        lines.append("  ⚠ RSI overbought (>70) — pullback risk")
    if rsi < 30:
        lines.append("  ⚠ RSI oversold (<30) — may drop further before reversal")
    if vol > 0.40:
        lines.append(f"  ⚠ High annualized volatility ({vol:.0%}) — wider stop required")
    if abs(beta) > 1.5:
        lines.append(f"  ⚠ High beta ({beta:.2f}) — amplified market moves")
    if not triggered:
        lines.append("  ⚠ No strategy triggered — lower conviction setup")

    return "\n".join(lines)


def generate_all_insights(ranked_stocks: list[dict], ml_predictions: dict | None = None) -> list[str]:
    insights = []
    for entry in ranked_stocks:
        sym = entry["symbol"]
        ml = ml_predictions.get(sym) if ml_predictions else None
        insights.append(generate_insight(sym, entry, ml_score=ml))
    return insights
