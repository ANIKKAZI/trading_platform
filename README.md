# Quantitative Stock Intelligence Platform

A production-grade, modular quantitative trading intelligence system. Analyses 5+ years of historical stock data, computes 30+ technical indicators, detects support/resistance zones, runs four pluggable strategy modules, trains and applies an XGBoost ML prediction layer via a built-in UI trainer, and outputs a daily ranked Top-N watchlist with explainable signals � all accessible through a unified visual dashboard, a standalone comparison interface, and a headless CLI.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Running the Platform](#running-the-platform)
4. [Interface Guide](#interface-guide)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Core Engines](#core-engines)
7. [Strategy Modules](#strategy-modules)
8. [Comparison & Forecast Engines](#comparison--forecast-engines)
9. [Configuration Reference](#configuration-reference)
10. [Output Formats](#output-formats)
11. [Adding a Custom Strategy](#adding-a-custom-strategy)
12. [Enabling ML Predictions](#enabling-ml-predictions)
13. [Disclaimer](#disclaimer)

---

## Project Structure

```
trading_model/
+-- main.py                          # CLI entry point for daily pipeline
+-- orchestrator.py                  # Full pipeline orchestrator
+-- requirements.txt
�
+-- config/
�   +-- settings.py                  # All tunable parameters
�   +-- app.py                       # Original scanner-only Streamlit app
�
+-- core/
�   +-- data_engine.py               # OHLCV fetching + parquet cache
�   +-- feature_engine.py            # 30+ technical indicators
�   +-- sr_engine.py                 # Support & resistance zone detection
�   +-- universe.py                  # Symbol universe loader
�
+-- strategies/
�   +-- base_strategy.py             # Abstract base class
�   +-- momentum_strategy.py         # EMA alignment + multi-period momentum
�   +-- mean_reversion_strategy.py   # RSI + Bollinger Band reversion
�   +-- breakout_strategy.py         # Resistance breakout + volume confirm
�   +-- volatility_strategy.py       # ATR expansion + BB squeeze
�
+-- engine/
�   +-- strategy_manager.py          # Plugin registry + weighted aggregation
�   +-- ranking_engine.py            # Composite scoring + Top-N selection
�   +-- ml_engine.py                 # Rule-based (Phase 1) + XGBoost (Phase 2)
�   +-- insight_engine.py            # Human-readable explanation generator
�   +-- output_formatter.py          # JSON serialization + file output
�   +-- market_context_engine.py     # Beta, correlation, sector, SPY trend
�   +-- scenario_engine.py           # Bull / Base / Bear probabilistic scenarios
�   +-- forecast_engine.py           # Full forecast orchestrator
�   +-- comparison_engine.py         # Side-by-side stock scoring + winner
�
+-- interfaces/
�   +-- compare_interface.py         # Programmatic comparison entry point
�
+-- ui/
�   +-- app.py                       # Unified dashboard (both tabs in one app)
�   +-- compare_dashboard.py         # Comparison-only Streamlit dashboard
�
+-- data/
�   +-- cache/                       # Parquet price cache (auto-created)
+-- output/                          # Daily JSON results (auto-created)
+-- models/                          # XGBoost model files (optional)
```

---

## Installation

### Prerequisites

- Python 3.10 or newer
- pip

### Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit plotly
```

Full dependency list:

| Package | Purpose |
|---|---|
| `yfinance` | OHLCV market data |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `pyarrow` | Parquet cache storage |
| `scikit-learn` | ML preprocessing |
| `xgboost` | Optional ML model |
| `joblib` | Model persistence |
| `lxml` / `html5lib` | S&P 500 Wikipedia scrape |
| `requests` | HTTP client |
| `streamlit` | Web UI framework |
| `plotly` | Interactive charts |

---

## Running the Platform

All commands assume the project root (`c:\Projects\trading_model`) as the working directory.

### Unified Dashboard (recommended)

Both the Daily Scanner and Stock Comparison are available in a single Streamlit app with two tabs:

```bash
streamlit run ui/app.py
```

Opens automatically at **http://localhost:8501** in your browser.

---

### Daily Scanner Dashboard (standalone)

```bash
streamlit run config/app.py
```

---

### Stock Comparison Dashboard (standalone)

```bash
streamlit run ui/compare_dashboard.py
```

---

### CLI � Daily Pipeline

```bash
# Default custom universe (defined in settings.py)
python main.py

# Full S&P 500
python main.py --universe sp500

# Specific tickers
python main.py --universe AAPL,MSFT,NVDA,GOOGL

# Options
python main.py --top 5              # Return top 5 instead of 10
python main.py --refresh            # Force re-download all price data
python main.py --quiet --no-save    # Silent mode, skip JSON output
python main.py --log-level DEBUG    # Verbose logging
```

---

### CLI � Stock Comparison

```bash
# Basic comparison (defaults to 5-month horizon)
python -m interfaces.compare_interface NVDA AAPL

# Custom forecast horizon in months
python -m interfaces.compare_interface NVDA AAPL --horizon 3

# Force refresh market data
python -m interfaces.compare_interface NVDA AAPL --refresh

# Skip saving output to file
python -m interfaces.compare_interface NVDA AAPL --no-save
```

Results are printed to the console and saved to `output/comparison_results_YYYY-MM-DD.json`.

---

## Interface Guide
### ML Settings (sidebar — global)

The very top of the sidebar applies to all three tabs:

| Control | Description |
|---|---|
| Enable ML Predictions (toggle) | **ON** → uses the trained XGBoost model for signal scoring; **OFF** → rule-based scoring only |

Status indicators:
- `✅ ML model active` — model file found and toggle is ON
- `⚠️ No model yet` — toggle ON but no model trained yet; go to **ML Training** tab
- `ℹ️ Rule-based scoring active` — toggle OFF

---
### Daily Scanner Tab

**Sidebar controls:**

| Control | Description |
|---|---|
| Universe Selection | `sp500` � full S&P 500; `custom` � symbols from `settings.py`; or comma-separated tickers |
| Top Ranked Stocks | Slider 3�20, controls how many results to return |
| Force Data Refresh (Scanner) | Re-downloads price data even if cache exists |
| Execute Quant Pipeline | Runs the full pipeline |

**Main area:**

- **Strategy Leaderboard** � ranked table of top-N stocks with composite score and signal label
- **Technical Chart Profile** � interactive candlestick chart (3-month window) with Support & Resistance zones overlaid
- **Ticker selector** � switch the chart to any stock in the leaderboard without re-running the pipeline
- **Signal Details** � score, signal, strategy score, and momentum score for the selected stock

---

### Stock Comparison Tab

**Sidebar controls:**

| Control | Description |
|---|---|
| Stock A / Stock B | Ticker symbols to compare (e.g. `NVDA`, `AAPL`) |
| Quick-pick tickers | Expander with 18 popular tickers |
| Forecast Horizon | 1 / 3 / 5 / 12 months |
| Scenario Sensitivity | Conservative / Balanced / Aggressive |
| Force Data Refresh (Compare) | Re-downloads price data for both stocks |
| Run Comparison | Runs the full comparison + forecast pipeline |

**Sensitivity modes:**

| Mode | Bull multiplier | Base multiplier | Bear multiplier |
|---|---|---|---|
| Conservative | x0.65 | x0.85 | x1.35 |
| Balanced | x1.00 | x1.00 | x1.00 |
| Aggressive | x1.45 | x1.15 | x0.65 |

**After running:**

- **Winner Banner** � which stock leads, confidence score, and active settings
- **Metric Cards** (7 per stock) � Price, Base Return, Bull Case, Bear Case, Volatility, Signal, Confidence
- **Six chart tabs:**

| Tab | Chart |
|---|---|
| Projection | 252-day historical % return + dashed forecast paths (bull/base/bear) with confidence bands |
| Risk / Reward | Scatter: risk score vs base expected return |
| Strategy Radar | Polar chart: Momentum / Mean Reversion / Breakout / Volatility / ML Confidence |
| S/R Zones | Side-by-side Support & Resistance zone charts with current price line |
| Monthly Trend | Month-by-month price projection with +-1 sigma band and bull/bear bounds |
| Distribution | Normal return distribution curves at the forecast horizon |

- **Key Factors** � top differentiating factors from the comparison engine
- **Signal Intelligence** � colour-coded badge chips for each strategy signal, trend direction, and risk flags
- **Full Comparison Scorecard** � expandable metric-by-metric breakdown

---
### ML Training Tab

Access via the **🤖 ML Training** tab in the unified dashboard.

**Model status panel** (top of tab):

| Field | Description |
|---|---|
| Model Status | ✅ Trained / ❌ Not Trained |
| Last Trained | Timestamp of last `models/xgb_model.pkl` write |
| Model Size | File size in KB |

**Training controls:**

| Control | Description |
|---|---|
| Training Symbols | Comma-separated tickers to train on (defaults to `CUSTOM_SYMBOLS` from `settings.py`) |
| Label Horizon | 5–60 trading days ahead used to create the binary up/down label (default 20d ≈ 1 month) |
| Force Data Refresh | Re-downloads price data before training |
| Train Model | Fetches data → computes features → trains → saves model |

**After training:**

- Metrics row: model type, accuracy %, train samples, test samples, symbols used
- Accuracy guide: **> 60%** strong edge · **55–60%** modest edge · **< 55%** near random
- Feature importance bar chart (horizontal, sorted by impact)
- Model saved automatically to `models/xgb_model.pkl`
- ML toggle in sidebar activates immediately for subsequent pipeline runs

**Accuracy guidance:**

Use **Force Data Refresh** + re-run the scanner or comparison after training a new model to ensure the latest model is used (the Streamlit cache is keyed by the ML toggle state).

---
## Pipeline Architecture

### Daily Scanner Pipeline

```
Universe loader
      |
      v
DataEngine --------- yfinance OHLCV + parquet cache
      |
      v
FeatureEngine ------- 30+ technical indicators
      |
      v
SREngine ------------ swing-based S/R zone detection
      |
      v
StrategyManager ----- 4 strategies x weighted signals -> final score
      |
      v
MLEngine ------------ rule-based bias (Phase 1) or XGBoost (Phase 2)
      |
      v
RankingEngine ------- composite score -> Top-N
      |
      v
InsightEngine ------- human-readable explanations
      |
      v
OutputFormatter ----- JSON file output
```

### Comparison & Forecast Pipeline

```
DataEngine (both stocks + SPY benchmark)
      |
      v
FeatureEngine + SREngine + StrategyManager + MLEngine
      |
      v
ForecastEngine
  |-- MarketContextEngine  (beta, volatility, sector, SPY trend)
  +-- ScenarioEngine       (bull/base/bear probabilities + price ranges)
        |
        v
ComparisonEngine --- 8-metric side-by-side scoring -> winner declaration
        |
        v
InsightEngine ------ comparison-specific narrative + risk factors
        |
        v
OutputFormatter ---- comparison JSON output
```

---

## Core Engines

### data_engine.py
- Fetches OHLCV data via yfinance
- Caches each symbol as a `.parquet` file in `data/cache/`
- Cache is considered stale after 3 days; `force_refresh=True` bypasses it
- `fetch_symbol(symbol, force_refresh) -> pd.DataFrame`
- `fetch_universe(symbols, force_refresh) -> dict[str, pd.DataFrame]`

### feature_engine.py
Computes 30+ features per row:

| Category | Features |
|---|---|
| Trend | EMA_20, EMA_50, EMA_200, Trend_direction, Trend_strength |
| Momentum | Mom_5d, Mom_20d, Mom_60d |
| Oscillators | RSI_14, MACD, MACD_signal, MACD_hist |
| Volatility | ATR, ATR_pct, Volatility, BB_upper, BB_lower, BB_mid, BB_pct, BB_width |
| Volume | Volume_ratio |
| Market | Beta, SPY correlation |

### sr_engine.py
- Detects swing highs/lows using a configurable window (`SR_SWING_WINDOW = 10`)
- Clusters nearby levels within `SR_CLUSTER_TOLERANCE` (default 2%)
- Requires at least `SR_MIN_TOUCHES = 2` touches to confirm a zone
- Returns `SRZones` with `support` and `resistance` as lists of (lo, hi) tuples

### universe.py
- `"sp500"` � scrapes current S&P 500 constituents from Wikipedia
- `"custom"` � uses `CUSTOM_SYMBOLS` from `settings.py`
- Any comma-separated string � treated as an ad-hoc ticker list

---

## Strategy Modules

Each strategy returns a signal (`+1` buy / `0` neutral / `-1` sell) and a confidence score (0.0�1.0).

| Strategy | Weight | Logic |
|---|---|---|
| MomentumStrategy | 30% | EMA alignment (20/50/200) + multi-period momentum (5d/20d/60d) |
| MeanReversionStrategy | 25% | RSI extremes (<30 / >70) + Bollinger Band outer touches |
| BreakoutStrategy | 25% | Price breaking resistance zone + volume > 1.5x average |
| VolatilityStrategy | 20% | Bollinger Band squeeze + ATR expansion confirmation |

**Composite signal formula:**

```
final_score = sum(signal x confidence x weight) / sum(weights) x 100
```

**Signal labels:**

| Score | Label |
|---|---|
| >= 60 | STRONG BUY |
| >= 25 | BUY |
| > -25 | NEUTRAL |
| <= -25 | SELL |
| <= -60 | STRONG SELL |

---

## Comparison & Forecast Engines

### market_context_engine.py
Builds macro context for a symbol:
- `beta` � relative movement vs SPY benchmark
- `volatility` � annualised historical volatility
- `spy_correlation` � Pearson correlation of daily returns with SPY
- `spy_trend` � bullish / bearish / neutral based on SPY EMA 20/50 cross
- `trend_direction` � strong_uptrend / uptrend / mixed / downtrend / strong_downtrend
- `sector` � mapped from built-in sector lookup table

### scenario_engine.py
Generates probabilistic scenarios over a configurable horizon:
- Horizon volatility = `ann_vol / sqrt(12) x sqrt(horizon_months)`
- Direction bias = `strategy_bias x 0.70 + ml_bias x 0.30`
- Returns bull_case, base_case, bear_case each with probability, expected_move, expected_move_percent, price_range

### forecast_engine.py
Orchestrates MarketContextEngine + ScenarioEngine into a complete forecast package, also producing:
- `risk_factors` � list of plain-language risk warnings
- `trend_summary` � one-line trend narrative

### comparison_engine.py
Scores two stocks across 8 weighted metrics and declares a winner:

| Metric | Weight |
|---|---|
| Base expected move | 25% |
| Bull expected move | 15% |
| Bear expected move (lower is better) | 15% |
| Volatility (risk-adjusted) | 10% |
| Strategy composite score | 15% |
| ML confidence | 5% |
| Trend score | 10% |
| Risk factor count | 5% |

Returns `winner`, `confidence` (0�100), `score_a`, `score_b`, `comparison_summary`, `key_factors`.

### ranking_engine.py
Composite score formula for the daily scanner:

```
composite = strategy_score x 0.45
          + momentum_score  x 0.25
          + breakout_prob   x 0.15
          + vol_adj_score   x 0.10
          + ml_score        x 0.05
```

---

## Configuration Reference

All settings live in `config/settings.py`:

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_UNIVERSE` | `"sp500"` | Default universe: `"sp500"` or `"custom"` |
| `CUSTOM_SYMBOLS` | 10 large-caps | Symbols used when universe = `"custom"` |
| `HISTORICAL_YEARS` | `5` | Years of OHLCV history to fetch |
| `BENCHMARK_SYMBOL` | `"SPY"` | Benchmark for beta and correlation |
| `TOP_N` | `10` | Number of top stocks to return |
| `ML_ENABLED` | `True` | Enable XGBoost ML layer (falls back to rule-based if no model file exists) |
| `ML_MODEL_PATH` | `"models/xgb_model.pkl"` | Path to trained model file |
| `EMA_PERIODS` | `[20, 50, 200]` | EMA windows |
| `RSI_PERIOD` | `14` | RSI lookback |
| `MACD_FAST / SLOW / SIGNAL` | `12 / 26 / 9` | MACD parameters |
| `ATR_PERIOD` | `14` | ATR window |
| `BOLLINGER_PERIOD` | `20` | Bollinger Band window |
| `BOLLINGER_STD` | `2` | Bollinger Band standard deviations |
| `MOMENTUM_PERIODS` | `[5, 20, 60]` | Momentum lookback periods in days |
| `SR_SWING_WINDOW` | `10` | Bars each side for swing detection |
| `SR_CLUSTER_TOLERANCE` | `0.02` | 2% price tolerance for S/R clustering |
| `SR_MIN_TOUCHES` | `2` | Minimum touches to confirm an S/R zone |
| `STRATEGY_WEIGHTS` | See file | Per-strategy weights dict |
| `OUTPUT_DIR` | `"output"` | Directory for JSON results |

---

## Output Formats

### Daily Scanner � `output/results_YYYY-MM-DD.json`

```json
{
  "date": "2026-05-22",
  "top_10_stocks": [
    {
      "symbol": "NVDA",
      "score": 93.4,
      "signal": "STRONG BUY",
      "support": [880.0, 895.0],
      "resistance": [930.0, 945.0],
      "strategies_triggered": ["MomentumStrategy", "BreakoutStrategy"],
      "prediction": "+2.1%",
      "strategy_score": 78.5,
      "momentum_score": 100.0
    }
  ]
}
```

### Stock Comparison � `output/comparison_results_YYYY-MM-DD.json`

```json
{
  "winner": "NVDA",
  "confidence": 72,
  "score_a": 68.4,
  "score_b": 41.2,
  "comparison_summary": ["NVDA leads on momentum...", "..."],
  "key_factors": ["Superior trend strength", "Higher ML confidence"],
  "detailed_forecast": {
    "NVDA": {
      "current_price": 1050.0,
      "horizon_months": 5,
      "scenarios": {
        "bull_case": { "probability": 0.30, "expected_move": "+28%", "price_range": [1200, 1380] },
        "base_case": { "probability": 0.50, "expected_move": "+12%", "price_range": [1100, 1220] },
        "bear_case": { "probability": 0.20, "expected_move": "-8%",  "price_range": [920, 1000] }
      },
      "market_context": { "beta": 1.72, "volatility": 0.42, "sector": "Technology" },
      "risk_factors": ["High beta relative to market"],
      "trend_summary": "Strong uptrend with EMA alignment"
    }
  }
}
```

---

## Adding a Custom Strategy

1. Create a new file in `strategies/`:

```python
from strategies.base_strategy import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "MyStrategy"

    def generate_signal(self, data: pd.DataFrame, context: dict) -> int:
        # Return +1 (buy), 0 (neutral), or -1 (sell)
        return 1

    def confidence_score(self, data: pd.DataFrame, context: dict) -> float:
        # Return a value between 0.0 and 1.0
        return 0.75
```

2. Register it in `orchestrator.py` inside `_build_strategy_manager()`:

```python
manager.register(MyStrategy(), weight=0.20)
```

No other changes required � the engine discovers and applies it automatically.

---

## Enabling ML Predictions

ML is **enabled by default** (`ML_ENABLED = True` in `config/settings.py`). When no trained model exists it automatically falls back to rule-based scoring — no crashes, no configuration needed.

### Recommended: train via the UI

1. Run `streamlit run ui/app.py`
2. Click the **🤖 ML Training** tab
3. Confirm or edit the training symbols (defaults to `CUSTOM_SYMBOLS`)
4. Set the label horizon (default 20 trading days ≈ 1 month)
5. Click **🚀 Train Model**
6. Watch the real-time progress bar; results and feature importances appear automatically
7. The **Enable ML Predictions** toggle in the sidebar (already ON) activates the model immediately

### UI toggle

The **Enable ML Predictions** toggle at the top of the sidebar lets you switch between the trained model and rule-based scoring at any time without restarting the app. Changing the toggle busts the Streamlit comparison cache so results always match the selected mode.

### Runtime override API

You can also control the ML state programmatically:

```python
from engine.ml_engine import set_runtime_ml_enabled, _is_ml_enabled

set_runtime_ml_enabled(True)   # activate XGBoost model
set_runtime_ml_enabled(False)  # force rule-based
print(_is_ml_enabled())        # check current effective state
```

### Training details

| Item | Detail |
|---|---|
| Model | XGBoost `XGBClassifier` (falls back to sklearn `GradientBoostingClassifier` if XGBoost not installed) |
| Task | Binary classification: price higher N days from now (1) or lower (0) |
| Features | 13 columns: `RSI, MACD, MACD_hist, ATR_pct, BB_pct, BB_width, Volume_ratio, Mom_5d, Mom_20d, Mom_60d, Volatility, Trend_strength, Beta` |
| Train/test split | 80 / 20, stratified |
| Save path | `models/xgb_model.pkl` (joblib pickle) |
| XGBoost params | 200 estimators · depth 4 · lr 0.05 · subsample 0.8 |

### Manual training (advanced)

If you prefer to train externally and plug in your own model:

```python
import joblib

# Your model must expose predict_proba(X) returning [p_down, p_up]
joblib.dump(trained_model, "models/xgb_model.pkl")
```

The prediction engine maps `predict_proba(X)[0][1]` → `[-1.0, +1.0]` using `prob_up * 2 - 1`.

---

## Disclaimer

This platform is designed for **research and educational purposes only**. It does not constitute financial advice. All trading involves significant risk of loss. Past performance does not guarantee future results. Always conduct your own due diligence before making any investment decisions.
