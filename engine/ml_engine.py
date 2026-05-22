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

# ---------------------------------------------------------------------------
# Runtime state — overrides ML_ENABLED from config at runtime (set by UI)
# ---------------------------------------------------------------------------

_runtime_ml_enabled: bool | None = None
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_runtime_ml_enabled(val: bool) -> None:
    """Override ML_ENABLED at runtime (called by the UI ML toggle)."""
    global _runtime_ml_enabled
    _runtime_ml_enabled = val


def _is_ml_enabled() -> bool:
    """Returns the effective ML enabled state (runtime override > config default)."""
    if _runtime_ml_enabled is not None:
        return _runtime_ml_enabled
    return ML_ENABLED


def _resolve_model_path(path: str) -> str:
    """Resolve a relative model path to absolute using project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


# ---------------------------------------------------------------------------
# Feature columns (must match FeatureEngine output)
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "RSI", "MACD", "MACD_hist", "ATR_pct", "BB_pct", "BB_width",
    "Volume_ratio", "Mom_5d", "Mom_20d", "Mom_60d",
    "Volatility", "Trend_strength", "Beta",
]


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------

class MLPredictionEngine:
    """
    Phase 1: rule-based scoring (always available).
    Phase 2: XGBoost model (loaded from disk when _is_ml_enabled() is True).
    """

    def __init__(self):
        self._model = None
        if _is_ml_enabled():
            self._load_model()

    def _load_model(self):
        path = _resolve_model_path(ML_MODEL_PATH)
        if not os.path.exists(path):
            logger.warning("ML model not found at %s. Falling back to rule-based.", path)
            return
        try:
            import joblib
            self._model = joblib.load(path)
            logger.info("XGBoost model loaded from %s", path)
        except Exception as exc:
            logger.error("Failed to load ML model: %s", exc)

    def predict(self, data: pd.DataFrame) -> float:
        """
        Returns a score in [-1.0, +1.0]:
            +1.0  → strong bullish prediction
            -1.0  → strong bearish prediction
        """
        if _is_ml_enabled():
            if self._model is None:
                # Lazy-load if toggle was enabled after __init__
                self._load_model()
            if self._model is not None:
                return self._xgb_predict(data)
        return self._rule_based_predict(data)

    def _rule_based_predict(self, data: pd.DataFrame) -> float:
        row = data.iloc[-1]
        score = 0.0
        total = 0.0

        # RSI signal
        rsi = row.get("RSI", 50)
        score += (rsi - 50) / 50
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
        features = [row.get(col, 0.0) for col in _FEATURE_COLS]
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


# ---------------------------------------------------------------------------
# ML Trainer
# ---------------------------------------------------------------------------

class MLTrainer:
    """
    Trains a binary direction classifier on feature-engineered OHLCV data.

    Labels:  1 if Close N days forward > Close today, else 0.
    Model:   XGBoost (preferred) or sklearn GradientBoostingClassifier (fallback).
    Saves:   joblib pickle to models/xgb_model.pkl (or custom path).
    """

    FEATURE_COLS = _FEATURE_COLS

    def train(
        self,
        symbol_data: dict[str, pd.DataFrame],
        label_horizon: int = 20,
        progress_callback=None,
        model_path: str | None = None,
    ) -> dict:
        """
        Train on feature-enriched DataFrames and save model to disk.

        Parameters
        ----------
        symbol_data      : {symbol: feature_enriched_df} — output of FeatureEngine
        label_horizon    : trading days ahead used to create the binary label
        progress_callback: callable(pct: float, msg: str) for UI progress updates
        model_path       : override save path; defaults to resolved ML_MODEL_PATH

        Returns
        -------
        dict with keys: model_type, accuracy, n_samples, n_train, n_test,
                        n_symbols, label_horizon_days, feature_importances, model_path
        On failure: dict with key 'error' containing a human-readable message.
        """
        if model_path is None:
            model_path = _resolve_model_path(ML_MODEL_PATH)

        def _cb(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(min(float(pct), 1.0), msg)
            else:
                logger.info("[%.0f%%] %s", pct * 100, msg)

        _cb(0.05, "Building training dataset…")

        frames: list[pd.DataFrame] = []
        for sym, df in symbol_data.items():
            if df.empty or len(df) < label_horizon + 50:
                logger.debug("Skipping %s — not enough rows (%d)", sym, len(df))
                continue
            missing = [c for c in self.FEATURE_COLS if c not in df.columns]
            if missing:
                logger.debug("Skipping %s — missing features: %s", sym, missing)
                continue
            chunk = df[self.FEATURE_COLS].copy()
            chunk["_label"] = (df["Close"].shift(-label_horizon) > df["Close"]).astype(int)
            chunk = chunk.dropna()
            if len(chunk) >= 20:
                frames.append(chunk)

        if not frames:
            return {
                "error": (
                    "No valid training data found. "
                    "Run the pipeline on some symbols first to populate feature data, "
                    "or select different symbols."
                )
            }

        combined = pd.concat(frames, ignore_index=True)
        X = combined[self.FEATURE_COLS].values
        y = combined["_label"].values

        _cb(0.25, f"Dataset ready: {len(X):,} samples from {len(frames)} symbols.")

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        _cb(0.40, "Training model…")

        # Try XGBoost first, fall back to sklearn
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            model_type = "XGBoost"
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
            )
            model_type = "GradientBoosting (sklearn)"

        model.fit(X_train, y_train)
        _cb(0.80, f"Evaluating {model_type} model…")

        from sklearn.metrics import accuracy_score
        accuracy = float(accuracy_score(y_test, model.predict(X_test)))

        importances: dict = {}
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(self.FEATURE_COLS, model.feature_importances_.tolist()))

        # Save model
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        import joblib
        joblib.dump(model, model_path)

        _cb(1.0, f"Model saved → {model_path}")

        return {
            "model_type": model_type,
            "accuracy": round(accuracy, 4),
            "n_samples": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_symbols": int(len(frames)),
            "label_horizon_days": label_horizon,
            "feature_importances": importances,
            "model_path": model_path,
        }
