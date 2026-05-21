# Quantitative Stock Intelligence Platform

A production-grade, modular quantitative trading intelligence system. Analyzes 5+ years of historical stock data, computes technical indicators, detects support/resistance zones, runs pluggable strategy modules, and outputs a daily ranked TOP 10 watchlist with explainable signals.

---

## Project Structure

```
trading_model/
├── main.py                      # CLI entry point
├── orchestrator.py              # Full pipeline orchestrator
├── requirements.txt
├── config/
│   └── settings.py              # All tunable parameters
├── core/
│   ├── data_engine.py           # OHLCV fetching + parquet cache
│   ├── feature_engine.py        # Technical indicators + market features
│   ├── sr_engine.py             # Support & resistance zone detection
│   └── universe.py              # Symbol universe loader (S&P 500 / custom)
├── strategies/
│   ├── base_strategy.py         # Abstract base class
│   ├── momentum_strategy.py     # EMA alignment + multi-period momentum
│   ├── mean_reversion_strategy.py  # RSI + Bollinger Band reversion
│   ├── breakout_strategy.py     # Resistance breakout + volume confirm
│   └── volatility_strategy.py  # ATR expansion + BB squeeze
├── engine/
│   ├── strategy_manager.py      # Dynamic plugin registry + aggregation
│   ├── ranking_engine.py        # Composite scoring + top-N selection
│   ├── ml_engine.py             # Rule-based (Phase 1) + XGBoost (Phase 2)
│   ├── insight_engine.py        # Human-readable explanation generator
│   └── output_formatter.py      # JSON serialization
├── data/
│   └── cache/                   # Parquet cache (auto-created)
├── output/                      # Daily JSON results (auto-created)
└── models/                      # XGBoost model files (optional)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with default universe (custom list from settings.py)

```bash
python main.py
```

### 3. Run on full S&P 500

```bash
python main.py --universe sp500
```

### 4. Run on specific tickers

```bash
python main.py --universe AAPL,MSFT,NVDA,GOOGL,AMZN
```

### 5. Additional options

```bash
python main.py --top 5 --refresh          # top 5, force refresh all data
python main.py --quiet --no-save          # silent mode, no file output
python main.py --log-level DEBUG          # verbose logging
```

---

## Output Format

Results are saved to `output/results_YYYY-MM-DD.json`:

```json
{
  "date": "2026-05-21",
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

---

## Adding a Custom Strategy

1. Create a file in `strategies/`:

```python
from strategies.base_strategy import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "MyStrategy"

    def generate_signal(self, data: pd.DataFrame, context: dict) -> int:
        # Return +1, 0, or -1
        return 1

    def confidence_score(self, data: pd.DataFrame, context: dict) -> float:
        return 0.75
```

2. Register it in `orchestrator.py` → `_build_strategy_manager()`:

```python
manager.register(MyStrategy(), weight=0.20)
```

That's it — no changes to the core engine required.

---

## Enabling ML Predictions (Phase 2)

1. Train and save your XGBoost model:

```python
import joblib
joblib.dump(trained_model, "models/xgb_model.pkl")
```

2. Set `ML_ENABLED = True` in `config/settings.py`.

The model should implement `predict_proba(X)` where index 1 = probability of upward move.

---

## Configuration

All parameters live in `config/settings.py`:

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_UNIVERSE` | `"custom"` | `"sp500"` or `"custom"` |
| `HISTORICAL_YEARS` | `5` | Years of history to fetch |
| `TOP_N` | `10` | Number of top stocks to return |
| `ML_ENABLED` | `False` | Enable XGBoost ML layer |
| `STRATEGY_WEIGHTS` | See file | Per-strategy signal weights |

---

## Disclaimer

This system is designed for **research and educational purposes only**. It does not constitute financial advice. All trading involves risk. Past performance does not guarantee future results.
