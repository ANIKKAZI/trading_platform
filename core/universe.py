"""
Universe loader.
Returns the list of symbols to analyze.
Supports S&P 500 (scraped from Wikipedia) or a custom list.
"""

import logging
import pandas as pd
from config.settings import DEFAULT_UNIVERSE, CUSTOM_SYMBOLS

logger = logging.getLogger(__name__)

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def load_universe(universe: str | None = None) -> list[str]:
    """
    Returns a list of ticker symbols.

    universe : "sp500" | "custom" | None (uses settings default)
    """
    universe = universe or DEFAULT_UNIVERSE

    if universe == "custom":
        logger.info("Using custom universe (%d symbols)", len(CUSTOM_SYMBOLS))
        return list(CUSTOM_SYMBOLS)

    if universe == "sp500":
        try:
            tables = pd.read_html(_SP500_URL)
            df = tables[0]
            symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info("Loaded S&P 500 universe (%d symbols)", len(symbols))
            return symbols
        except Exception as exc:
            logger.warning("Failed to fetch S&P 500 list (%s). Falling back to custom list.", exc)
            return list(CUSTOM_SYMBOLS)

    # Treat as comma-separated or list
    if isinstance(universe, str) and "," in universe:
        return [s.strip().upper() for s in universe.split(",")]

    logger.warning("Unknown universe '%s'. Using custom symbols.", universe)
    return list(CUSTOM_SYMBOLS)
