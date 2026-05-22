# ML Prediction Layer — Technical Reference

This document covers everything about the machine learning layer in the Quantitative Stock Intelligence Platform: how it integrates with the Daily Scanner and Stock Comparison, what training does under the hood, and how to get the most out of it.

---

## Table of Contents

1. [Overview](#overview)
2. [How ML Integrates with the Daily Scanner](#how-ml-integrates-with-the-daily-scanner)
3. [How ML Integrates with the Stock Comparison](#how-ml-integrates-with-the-stock-comparison)
4. [What Training Actually Does](#what-training-actually-does)
5. [The UI Toggle](#the-ui-toggle)
6. [Training via the UI](#training-via-the-ui)
7. [Training Details](#training-details)
8. [Accuracy Guide](#accuracy-guide)
9. [Runtime API](#runtime-api)
10. [Manual / External Model](#manual--external-model)
11. [File Reference](#file-reference)

---

## Overview

The ML layer is a **pluggable binary direction classifier** that predicts whether a stock's price will be higher or lower N trading days from now. It sits on top of the four strategy modules and injects a data-driven signal into both the scanner ranking and the comparison forecast.

Two modes are always available:

| Mode | When active | How it works |
|---|---|---|
| **Rule-based (Phase 1)** | No trained model, or toggle OFF | Hand-coded formula using RSI, MACD histogram, momentum, trend strength |
| **XGBoost (Phase 2)** | Trained model exists AND toggle ON | Learned non-linear patterns across all 13 features simultaneously |

The system never crashes in rule-based mode — it is the safe default whenever no model file is present.

---

## How ML Integrates with the Daily Scanner

The scanner produces a **composite ranking score** for every stock in the universe:

```
composite = strategy_score × 0.45
          + momentum_score  × 0.25
          + breakout_prob   × 0.15
          + vol_adj_score   × 0.10
          + ml_score        × 0.05
```

The `ml_score` is the output of `MLPredictionEngine.predict(features_df)`, mapped to the range `[-1.0, +1.0]`.

**What this means in practice:**

- The ML layer carries a **5% weight** — it nudges scores rather than dominating them. A stock strong on strategy + momentum signals will not be knocked down by a neutral ML score alone.
- Two similarly-ranked stocks that are otherwise equal on strategy signals will be separated by the ML score. This is where the model adds the most differentiation.
- With a well-trained model (accuracy > 57%), stocks that have historically shown the feature patterns associated with upward moves will rank slightly higher than pure rule-based scoring would place them.

**Rule-based fallback formula (no model):**

```
score = (RSI - 50) / 50
      + sign(MACD_hist) × 0.5
      + clip(Mom_5d × 10, -1, 1) × 0.4
      + clip(Mom_20d × 5, -1, 1) × 0.6
      + clip(Trend_strength × 5, -1, 1)
      / total_weights
```

---

## How ML Integrates with the Stock Comparison

ML feeds into the comparison pipeline in **two distinct places**:

### 1. Scenario direction bias (`scenario_engine.py`)

Every forecast scenario (bull / base / bear) is built around a direction bias:

```
direction_bias = strategy_bias × 0.70 + ml_bias × 0.30
```

The `ml_bias` is the raw `[-1.0, +1.0]` ML score for that stock. This bias shifts the expected move percentages for all three scenarios up or down.

**Example:** If NVDA's XGBoost score is `+0.65` (strong bullish) and AAPL's is `+0.10` (barely positive), NVDA's base case projected return will be meaningfully higher than AAPL's, and this difference flows through to:
- The **Projection chart** (dashed forecast paths)
- The **Monthly Trend chart** (price projection month by month)
- The **Distribution chart** (probability distribution peak position)
- The **Metric Cards** (Base Return %, Bull Case %, Bear Case %)

### 2. Head-to-head comparison scoring (`comparison_engine.py`)

The comparison engine scores both stocks across 8 weighted metrics:

| Metric | Weight | ML connection |
|---|---|---|
| Base expected move | 25% | Indirectly — shaped by ML via direction_bias |
| Bull expected move | 15% | Indirectly — shaped by ML via direction_bias |
| Bear expected move | 15% | Indirectly — shaped by ML via direction_bias |
| Volatility (risk-adj) | 10% | — |
| Strategy composite score | 15% | — |
| **ML confidence** | **5%** | Direct — raw ML score used here |
| Trend score | 10% | — |
| Risk factor count | 5% | — |

The ML confidence score also appears as the 5th spoke on the **Strategy Radar chart** (labelled "ML Confidence"), giving a visual read of model-predicted upside for each stock side-by-side.

---

## What Training Actually Does

When you click **Train Model** in the UI, the following pipeline runs:

```
For each training symbol:
    1. Fetch 5 years of OHLCV via yfinance (cached as .parquet)
    2. Compute all 13 features via FeatureEngine
    3. Create binary label:
         label[t] = 1  if Close[t + horizon] > Close[t]
         label[t] = 0  if Close[t + horizon] ≤ Close[t]
    4. Drop rows with NaN (start/end of series)

Combine all symbols → shuffle → 80/20 train/test split (stratified)

Train XGBoost XGBClassifier:
    n_estimators=200, max_depth=4, learning_rate=0.05
    subsample=0.8, colsample_bytree=0.8

Evaluate on held-out test set → report accuracy
Save model → models/xgb_model.pkl
```

**What the model learns:**

XGBoost finds non-linear combinations of features that historically preceded upward price moves over your chosen horizon. For example, it might learn that:

- RSI between 45–55 AND MACD histogram turning positive AND Mom_20d > 0.02 is a stronger buy signal than any single indicator alone
- High BB_width (volatility expansion) combined with high Volume_ratio AND positive trend strength has historically resolved upward more often than not

These patterns are **stock-agnostic** — the model is trained on all symbols combined, so it generalises across different stocks rather than overfitting to one.

**Label horizon choice matters:**

| Horizon | What you're predicting | Best for |
|---|---|---|
| 5–10d | Very short-term momentum | Noisy, hard to predict |
| 20d (default) | ~1 month direction | Balanced signal quality |
| 30d | ~6 weeks direction | Smoother labels, less noise |
| 60d | ~3 months direction | Longer-term bias detection |

Shorter horizons produce noisier labels (market noise dominates) and typically lower accuracy. Longer horizons produce cleaner labels but the model is less reactive to current conditions. **20–30 days is generally the best starting point.**

---

## The UI Toggle

The **Enable ML Predictions** toggle at the top of the sidebar is a global runtime switch:

| State | Effect |
|---|---|
| **ON + model exists** | All scanner rankings and comparison forecasts use XGBoost predictions |
| **ON + no model** | Falls back to rule-based automatically, shows ⚠️ in sidebar |
| **OFF** | Rule-based scoring used everywhere, model file ignored |

**Practical use — comparing rule-based vs ML:**

1. Run the comparison or scanner with toggle **OFF** → note the scores
2. Switch toggle **ON** → click Force Data Refresh → run again
3. Differences between the two runs isolate exactly what the ML layer is contributing

If the results are nearly identical, either your model accuracy is low (near 50%) or the feature patterns in the current data are not strongly differentiated. If the ML-on results consistently separate strong setups from weak ones differently than rule-based, the model is adding genuine signal.

**Cache behaviour:** The Streamlit comparison cache is keyed by the toggle state (`ml_enabled` is a parameter to `_run_comparison`). Switching the toggle always triggers a fresh pipeline run rather than returning a stale cached result.

---

## Training via the UI

**Step-by-step:**

1. `streamlit run ui/app.py`
2. Navigate to the **🤖 ML Training** tab
3. Edit the training symbols if needed (more = better generalisation, but slower)
4. Set the label horizon (20d is a good default)
5. Optionally enable **Force Data Refresh** if your cached data is stale
6. Click **🚀 Train Model**
7. Watch the progress bar update through: dataset build → train/test split → model fit → evaluation → save
8. Review the accuracy and feature importance chart
9. The model is now active — the sidebar shows ✅ ML model active

**After training:** If you re-run the scanner or comparison immediately, use **Force Data Refresh** to ensure the Streamlit cache is cleared and the new model is used.

**Recommended training symbol sets:**

| Goal | Symbols to use |
|---|---|
| General market model | S&P 500 via the scanner (run full universe first to populate cache, then train) |
| Tech-focused model | NVDA, AAPL, MSFT, GOOGL, META, AMZN, TSLA, AMD, ORCL, CRM, NFLX, INTC |
| Balanced model | Mix of tech, finance, healthcare, energy (10–20 symbols minimum) |

More symbols = more training samples = better generalisation. Under ~5 symbols the model is likely to overfit.

---

## Training Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost `XGBClassifier` |
| Fallback | sklearn `GradientBoostingClassifier` (if XGBoost not installed) |
| Task | Binary classification (1 = up, 0 = down/flat) |
| Train/test split | 80 / 20, stratified by label |
| Estimators | 200 |
| Max depth | 4 |
| Learning rate | 0.05 |
| Subsample | 0.8 |
| Column subsample | 0.8 (colsample_bytree) |
| Eval metric | log-loss |
| Minimum rows per symbol | 100 (symbols with fewer rows are skipped) |
| Minimum label horizon | 5 days |
| Maximum label horizon | 60 days |
| Save format | joblib pickle |
| Save path | `models/xgb_model.pkl` |

**The 13 input features:**

| Feature | Description |
|---|---|
| `RSI` | 14-period Relative Strength Index |
| `MACD` | MACD line (EMA12 − EMA26) |
| `MACD_hist` | MACD histogram (MACD − signal) |
| `ATR_pct` | ATR as % of close price |
| `BB_pct` | Position within Bollinger Bands (0=lower, 1=upper) |
| `BB_width` | Bollinger Band width (upper − lower) / mid |
| `Volume_ratio` | Current volume / 20-day average volume |
| `Mom_5d` | 5-day price momentum |
| `Mom_20d` | 20-day price momentum |
| `Mom_60d` | 60-day price momentum |
| `Volatility` | 20-day annualised historical volatility |
| `Trend_strength` | Composite EMA alignment score |
| `Beta` | Beta relative to SPY benchmark |

---

## Accuracy Guide

| Accuracy | Interpretation | Recommended action |
|---|---|---|
| > 60% | Strong edge — model is finding real patterns | Use as-is; consider longer horizon |
| 55–60% | Modest edge — useful but not dominant | Add more symbols, try 25–30d horizon |
| 52–55% | Marginal — slight edge over random | Significantly expand symbol set |
| < 52% | Near random — model adds little | Retrain with more data or different horizon |

**Note:** Accuracies above 65% on financial data are rare and may indicate label leakage. 55–60% is a realistic and useful target.

---

## Runtime API

Control the ML state from code without touching `config/settings.py`:

```python
from engine.ml_engine import (
    set_runtime_ml_enabled,
    _is_ml_enabled,
    MLPredictionEngine,
    MLTrainer,
)

# Enable or disable at runtime
set_runtime_ml_enabled(True)
set_runtime_ml_enabled(False)

# Check current effective state (runtime override > config default)
print(_is_ml_enabled())

# Use the prediction engine directly
engine = MLPredictionEngine()
score = engine.predict(feature_df)   # returns float in [-1.0, +1.0]

# Train programmatically
trainer = MLTrainer()
result = trainer.train(
    symbol_data={"NVDA": nvda_features_df, "AAPL": aapl_features_df},
    label_horizon=20,
    progress_callback=lambda pct, msg: print(f"{pct:.0%} {msg}"),
)
print(result["accuracy"], result["feature_importances"])
```

---

## Manual / External Model

You can plug in any scikit-learn compatible model as long as it exposes `predict_proba(X)`:

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train your own model with the same 13 features in the same column order:
# RSI, MACD, MACD_hist, ATR_pct, BB_pct, BB_width, Volume_ratio,
# Mom_5d, Mom_20d, Mom_60d, Volatility, Trend_strength, Beta

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

joblib.dump(model, "models/xgb_model.pkl")
```

The prediction engine reads `predict_proba(X)[0][1]` (probability of class 1 = up) and maps it to `[-1.0, +1.0]` via `prob_up * 2 - 1`. Any binary classifier that exposes this interface will work.

---

## File Reference

| File | Role |
|---|---|
| `engine/ml_engine.py` | `MLPredictionEngine` (predict), `MLTrainer` (train), `set_runtime_ml_enabled`, `_is_ml_enabled` |
| `engine/scenario_engine.py` | Consumes ML score as 30% of direction_bias in scenario building |
| `engine/comparison_engine.py` | Uses ML confidence as one of 8 scored metrics |
| `engine/ranking_engine.py` | Uses ML score as 5% of composite scanner ranking |
| `config/settings.py` | `ML_ENABLED = True`, `ML_MODEL_PATH = "models/xgb_model.pkl"` |
| `models/xgb_model.pkl` | Trained model file (created by MLTrainer, gitignored) |
| `ui/app.py` | ML toggle in sidebar, ML Training tab (`_render_ml_tab`) |
