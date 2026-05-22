"""
Comparison Engine.
Side-by-side scoring of two fully-processed stock payloads.
Declares a winner and returns a structured comparison result with a
weighted metric scorecard, confidence score, and key winning factors.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Metric specification:
#   (display_label, internal_key, higher_is_better, weight)
_METRICS: list[tuple[str, str, bool, float]] = [
    ("Expected Return (Base Case)",  "base_expected_move",  True,  0.25),
    ("Upside Potential (Bull Case)", "bull_expected_move",  True,  0.15),
    ("Downside Risk (Bear Case)",    "bear_expected_move",  False, 0.15),
    ("Volatility",                   "volatility",          False, 0.10),
    ("Strategy Alignment Score",     "strategy_score",      True,  0.15),
    ("ML Prediction Confidence",     "ml_confidence",       True,  0.05),
    ("Trend Score",                  "trend_score",         True,  0.10),
    ("Risk Factor Count",            "risk_score",          False, 0.05),
]


class ComparisonEngine:
    """
    Compares two stock payloads built by CompareInterface and returns a
    winner declaration with scorecard, confidence, and key factors.

    Expected payload shape (each stock)
    ------------------------------------
    {
        symbol          : str,
        latest_row      : pd.Series,          # last row of feature DataFrame
        strategy_output : dict,               # from StrategyManager.aggregate()
        ml_output       : float,              # [-1.0, +1.0]
        scenarios       : dict,               # from ScenarioEngine
        market_context  : dict,               # from MarketContextEngine
        risk_factors    : list[str],
    }
    """

    def compare(self, stock_a: dict, stock_b: dict) -> dict:
        """
        Compare two stock payloads.

        Returns
        -------
        {
            winner              : str,
            confidence          : int (0–100),
            score_a             : int,
            score_b             : int,
            comparison_summary  : list[str],
            key_factors         : list[str],
            metric_scores       : {symbol_a: dict, symbol_b: dict},
        }
        """
        metrics_a = self._extract_metrics(stock_a)
        metrics_b = self._extract_metrics(stock_b)

        score_a = 0.0
        score_b = 0.0
        summary: list[str] = []

        for label, key, higher_is_better, weight in _METRICS:
            val_a = float(metrics_a.get(key, 0.0))
            val_b = float(metrics_b.get(key, 0.0))

            if abs(val_a - val_b) < 1e-9:
                summary.append(f"{label}: Tied ({val_a:.2f})")
                score_a += weight / 2
                score_b += weight / 2
                continue

            a_wins = (val_a > val_b) if higher_is_better else (val_a < val_b)
            winner_val, loser_val = (val_a, val_b) if a_wins else (val_b, val_a)
            winner_sym = stock_a["symbol"] if a_wins else stock_b["symbol"]

            if a_wins:
                score_a += weight
            else:
                score_b += weight

            direction = "higher" if higher_is_better else "lower"
            summary.append(
                f"{label}: {winner_sym} wins "
                f"({winner_val:.2f} vs {loser_val:.2f} — {direction} is better)"
            )

        total = score_a + score_b
        if total < 1e-9:
            winner_sym = stock_a["symbol"]
            confidence = 50
        elif score_a >= score_b:
            winner_sym = stock_a["symbol"]
            confidence = round((score_a / total) * 100)
        else:
            winner_sym = stock_b["symbol"]
            confidence = round((score_b / total) * 100)

        winner_payload = stock_a if winner_sym == stock_a["symbol"] else stock_b
        loser_payload = stock_b if winner_sym == stock_a["symbol"] else stock_a
        key_factors = self._build_key_factors(winner_payload, loser_payload)

        return {
            "winner": winner_sym,
            "confidence": confidence,
            "score_a": round(score_a * 100),
            "score_b": round(score_b * 100),
            "comparison_summary": summary,
            "key_factors": key_factors,
            "metric_scores": {
                stock_a["symbol"]: metrics_a,
                stock_b["symbol"]: metrics_b,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_metrics(self, payload: dict) -> dict[str, float]:
        """
        Flatten a stock payload into the scalar metrics used by the scorecard.
        Safe against missing keys — falls back to 0.0 for any absent value.
        """
        scenarios = payload.get("scenarios", {})
        context = payload.get("market_context", {})
        strategy_output = payload.get("strategy_output", {})
        ml_output = float(payload.get("ml_output", 0.0))
        row: pd.Series | None = payload.get("latest_row")

        base_move = float(
            scenarios.get("base_case", {}).get("expected_move_percent", 0.0)
        )
        bull_move = float(
            scenarios.get("bull_case", {}).get("expected_move_percent", 0.0)
        )
        bear_move = float(
            scenarios.get("bear_case", {}).get("expected_move_percent", 0.0)
        )

        volatility = float(context.get("volatility", 0.25))
        strategy_score = float(strategy_output.get("final_score", 0.0))
        ml_confidence = (ml_output + 1.0) / 2.0  # remap [-1, 1] → [0, 1]

        # Trend score: 1 point per bullish indicator alignment
        trend_score = 0.0
        if row is not None:
            close = float(row.get("Close", 0.0))
            if close > float(row.get("EMA_20", float("inf"))):
                trend_score += 1
            if close > float(row.get("EMA_50", float("inf"))):
                trend_score += 1
            if close > float(row.get("EMA_200", float("inf"))):
                trend_score += 1
            if float(row.get("MACD_hist", 0.0)) > 0:
                trend_score += 1
            if float(row.get("Mom_20d", 0.0)) > 0:
                trend_score += 1

        risk_score = float(len(payload.get("risk_factors", [])))

        return {
            "base_expected_move": base_move,
            "bull_expected_move": bull_move,
            "bear_expected_move": bear_move,
            "volatility": volatility,
            "strategy_score": strategy_score,
            "ml_confidence": ml_confidence,
            "trend_score": trend_score,
            "risk_score": risk_score,
        }

    def _build_key_factors(
        self, winner: dict, loser: dict
    ) -> list[str]:
        """
        Return up to 5 concise sentences explaining why the winner leads.
        """
        factors: list[str] = []

        # Strategy score gap
        w_score = float(winner.get("strategy_output", {}).get("final_score", 0.0))
        l_score = float(loser.get("strategy_output", {}).get("final_score", 0.0))
        if w_score > l_score + 5:
            factors.append(
                f"Higher strategy aggregate score ({w_score:.1f} vs {l_score:.1f})"
            )

        # Extra strategy triggers
        w_triggered = set(winner.get("strategy_output", {}).get("triggered", []))
        l_triggered = set(loser.get("strategy_output", {}).get("triggered", []))
        extra = w_triggered - l_triggered
        if extra:
            factors.append(
                f"Additional strategy signals active: {', '.join(sorted(extra))}"
            )

        # EMA trend
        w_trend = winner.get("market_context", {}).get("trend_direction", "mixed")
        l_trend = loser.get("market_context", {}).get("trend_direction", "mixed")
        _strong = {"strong_uptrend", "uptrend"}
        _weak = {"downtrend", "strong_downtrend"}
        if w_trend in _strong and l_trend not in _strong:
            factors.append("Stronger EMA trend alignment (price above key moving averages)")

        # Base-case expected move
        w_base = float(
            winner.get("scenarios", {}).get("base_case", {}).get("expected_move_percent", 0.0)
        )
        l_base = float(
            loser.get("scenarios", {}).get("base_case", {}).get("expected_move_percent", 0.0)
        )
        if w_base > l_base + 1:
            factors.append(
                f"Better base-case projected return ({w_base:+.1f}% vs {l_base:+.1f}%)"
            )

        # Volatility risk-adjusted advantage
        w_vol = float(winner.get("market_context", {}).get("volatility", 0.3))
        l_vol = float(loser.get("market_context", {}).get("volatility", 0.3))
        if w_vol < l_vol - 0.05:
            factors.append(
                f"Lower volatility ({w_vol:.0%} vs {l_vol:.0%}) — better risk-adjusted profile"
            )

        if not factors:
            factors.append("Marginally higher composite score across multiple metrics")

        return factors[:5]
