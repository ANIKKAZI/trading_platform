"""
Global configuration for the Quantitative Stock Intelligence Platform.
"""

# Universe
DEFAULT_UNIVERSE = "sp500"  # or "custom"
CUSTOM_SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "UNH"]

# Data
HISTORICAL_YEARS = 5
DATA_PROVIDER = "yfinance"  # yfinance | alphavantage
BENCHMARK_SYMBOL = "SPY"
DATA_CACHE_DIR = "data/cache"

# Feature engineering
EMA_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
VOLUME_MA_PERIOD = 20
MOMENTUM_PERIODS = [5, 20, 60]
VOLATILITY_WINDOW = 20

# Support/Resistance
SR_SWING_WINDOW = 10        # bars each side for swing detection
SR_CLUSTER_TOLERANCE = 0.02  # 2% price tolerance for clustering
SR_MIN_TOUCHES = 2

# Strategy weights (must sum to 1.0)
STRATEGY_WEIGHTS = {
    "MomentumStrategy": 0.30,
    "MeanReversionStrategy": 0.25,
    "BreakoutStrategy": 0.25,
    "VolatilityStrategy": 0.20,
}

# Ranking
TOP_N = 10
ML_ENABLED = True           # enable XGBoost layer when model exists (toggle in UI)
ML_MODEL_PATH = "models/xgb_model.pkl"

# Output
OUTPUT_DIR = "output"
LOG_LEVEL = "INFO"
