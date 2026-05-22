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


# ---------------------------------------------------------------------------
# Comparison-specific insight generator (added for compare_interface)
# ---------------------------------------------------------------------------

def generate_comparison_insight(
    symbol: str,
    payload: dict,
    forecast: dict,
    comparison_result: dict,
) -> str:
    """
    Generate a human-readable comparison insight for one stock.

    Parameters
    ----------
    symbol              : ticker string
    payload             : comparison payload dict (includes latest_row, strategy_output, etc.)
    forecast            : forecast dict from ForecastEngine.forecast()
    comparison_result   : dict returned by ComparisonEngine.compare()

    Returns
    -------
    Multi-line string suitable for CLI display.
    """
    row: pd.Series = payload.get("latest_row")
    strategy_output: dict = payload.get("strategy_output", {})
    ml_output: float = float(payload.get("ml_output", 0.0))
    sr_zones = payload.get("sr_zones")
    scenarios: dict = forecast.get("scenarios", {})
    risk_factors: list[str] = forecast.get("risk_factors", [])
    context: dict = forecast.get("market_context", {})
    winner: str = comparison_result.get("winner", "")
    is_winner = winner == symbol

    lines: list[str] = [
        f"{'─' * 50}",
        f"  {symbol}{'  ← WINNER' if is_winner else ''}",
        f"  Current Price: ${forecast.get('current_price', 0):.2f}",
        f"{'─' * 50}",
    ]

    # Strategy signals
    raw_signals: dict = strategy_output.get("raw_signals", {})
    confidences: dict = strategy_output.get("confidences", {})
    triggered: list[str] = strategy_output.get("triggered", [])

    lines.append("Strategy Signals:")
    for name, sig in raw_signals.items():
        direction = "Bullish" if sig == 1 else ("Bearish" if sig == -1 else "Neutral")
        conf = confidences.get(name, 0.5)
        lines.append(f"  {name:<28} {direction:<10}  (conf: {conf:.0%})")

    # 5-month scenarios
    lines.append(f"\n{forecast.get('horizon_months', 5)}-Month Forecast Scenarios:")
    for case_key, label in [
        ("bull_case", "Bull"), ("base_case", "Base"), ("bear_case", "Bear")
    ]:
        sc = scenarios.get(case_key, {})
        move = sc.get("expected_move", "N/A")
        prob = sc.get("probability", 0.0)
        pr = sc.get("price_range", [])
        pr_str = f"  ${pr[0]:.2f}–${pr[1]:.2f}" if len(pr) == 2 else ""
        lines.append(f"  {label:<6}: {move:<8}  (p={prob:.0%}){pr_str}")

    # Key technicals
    if row is not None:
        lines.append("\nKey Technicals:")
        lines.append(f"  Close:     ${float(row.get('Close', 0)):.2f}")
        lines.append(f"  RSI:       {float(row.get('RSI', float('nan'))):.1f}")
        lines.append(
            f"  EMA20: {float(row.get('EMA_20', float('nan'))):.2f}  "
            f"EMA50: {float(row.get('EMA_50', float('nan'))):.2f}  "
            f"EMA200: {float(row.get('EMA_200', float('nan'))):.2f}"
        )
        lines.append(f"  MACD hist: {float(row.get('MACD_hist', float('nan'))):.3f}")
        lines.append(f"  ATR%:      {float(row.get('ATR_pct', float('nan'))):.2%}")
        lines.append(
            f"  5D Mom: {float(row.get('Mom_5d', float('nan'))):.2%}  "
            f"20D: {float(row.get('Mom_20d', float('nan'))):.2%}  "
            f"60D: {float(row.get('Mom_60d', float('nan'))):.2%}"
        )
        lines.append(
            f"  Beta: {float(context.get('beta', float('nan'))):.2f}  "
            f"Corr(SPY): {float(context.get('spy_correlation', float('nan'))):.2f}  "
            f"Ann.Vol: {float(context.get('volatility', float('nan'))):.1%}"
        )

    # S/R zones
    if sr_zones is not None:
        if sr_zones.resistance:
            lo, hi = sr_zones.resistance[0]
            lines.append(f"\n  Nearest Resistance: {lo:.2f}–{hi:.2f}")
        if sr_zones.support:
            lo, hi = sr_zones.support[0]
            lines.append(f"  Nearest Support:    {lo:.2f}–{hi:.2f}")

    # ML prediction
    ml_pct = ml_output * 3
    direction_str = "positive" if ml_pct >= 0 else "negative"
    lines.append(f"\n  ML Prediction bias: {ml_pct:+.1f}% ({direction_str})")

    # Risk factors
    lines.append("\nRisk Factors:")
    for r in risk_factors:
        lines.append(f"  ⚠  {r}")

    # Winner key factors (only shown for the winning stock)
    if is_winner:
        key_factors: list[str] = comparison_result.get("key_factors", [])
        if key_factors:
            lines.append("\nWhy this stock ranked stronger:")
            for f in key_factors:
                lines.append(f"  ✓  {f}")

    return "\n".join(lines)
