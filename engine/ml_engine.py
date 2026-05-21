"""
ML Prediction Engine — Phase 1 (Rule-based) + Phase 2 (XGBoost).
Pluggable model layer that returns a directional prediction score per symbol.
"""

import logging
import os
import numpy as np
import pandas as pd

from config.settings import ML_ENABLED, ML_MODEL_PATH

logger = logging.getLogger(__name__)

_FEATURE_COLS = [
    "RSI", "MACD", "MACD_hist", "ATR_pct", "BB_pct", "BB_width",
    "Volume_ratio", "Mom_5d", "Mom_20d", "Mom_60d",
    "Volatility", "Trend_strength", "Beta",
]


class MLPredictionEngine:
    """
    Phase 1: rule-based scoring (always available).
    Phase 2: XGBoost model (loaded from disk if ML_ENABLED=True).
    """

    def __init__(self):
        self._model = None
        if ML_ENABLED:
            self._load_model()

    def _load_model(self):
        if not os.path.exists(ML_MODEL_PATH):
            logger.warning("ML model not found at %s. Falling back to rule-based.", ML_MODEL_PATH)
            return
        try:
            import joblib
            self._model = joblib.load(ML_MODEL_PATH)
            logger.info("XGBoost model loaded from %s", ML_MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load ML model: %s", exc)

    def predict(self, data: pd.DataFrame) -> float:
        """
        Returns a score in [-1.0, +1.0]:
            +1.0  → strong bullish prediction
            -1.0  → strong bearish prediction
        """
        if ML_ENABLED and self._model is not None:
            return self._xgb_predict(data)
        return self._rule_based_predict(data)

    def _rule_based_predict(self, data: pd.DataFrame) -> float:
        row = data.iloc[-1]
        score = 0.0
        total = 0.0

        # RSI signal
        rsi = row.get("RSI", 50)
        score += (50 - rsi) / 50 * -1   # RSI > 50 → positive
        score += (rsi - 50) / 50         # normalized
        total += 1

        # MACD histogram direction
        macd_hist = row.get("MACD_hist", 0)
        score += np.sign(macd_hist) * 0.5
        total += 1

        # Momentum
        mom_5 = row.get("Mom_5d", 0)
        mom_20 = row.get("Mom_20d", 0)
        score += np.clip(mom_5 * 10, -1, 1) * 0.4
        score += np.clip(mom_20 * 5, -1, 1) * 0.6
        total += 1

        # Trend strength
        ts = row.get("Trend_strength", 0)
        score += np.clip(ts * 5, -1, 1)
        total += 1

        return float(np.clip(score / total, -1.0, 1.0))

    def _xgb_predict(self, data: pd.DataFrame) -> float:
        row = data.iloc[-1]
        features = []
        for col in _FEATURE_COLS:
            features.append(row.get(col, 0.0))
        X = np.array(features).reshape(1, -1)
        try:
            prob = self._model.predict_proba(X)[0]
            # Assume binary: index 0=down, 1=up
            return float(prob[1] * 2 - 1)  # map [0,1] → [-1, +1]
        except Exception as exc:
            logger.warning("XGBoost predict failed: %s", exc)
            return self._rule_based_predict(data)


def predict_all(engine: MLPredictionEngine, symbol_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Run prediction on all symbols. Returns {symbol: score}."""
    return {sym: engine.predict(df) for sym, df in symbol_data.items()}
