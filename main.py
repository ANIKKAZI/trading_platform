"""
CLI entry point for the Quantitative Stock Intelligence Platform.

Usage:
    python main.py                          # default universe (settings.py)
    python main.py --universe custom        # CUSTOM_SYMBOLS from settings
    python main.py --universe sp500         # full S&P 500
    python main.py --universe AAPL,MSFT,NVDA  # ad-hoc symbols
    python main.py --top 5 --refresh        # top 5, force data refresh
    python main.py --quiet                  # suppress CLI output, save JSON only
"""

import argparse
import logging
import sys
from orchestrator import run
from config.settings import LOG_LEVEL, TOP_N


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative Stock Intelligence Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--universe",
        default=None,
        help="Universe to analyze: 'sp500' | 'custom' | comma-separated tickers",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_N,
        help="Number of top stocks to return",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download all market data",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save JSON output to disk",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed console output",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    args = parser.parse_args()
    _setup_logging(args.log_level)

    try:
        run(
            universe=args.universe,
            force_refresh=args.refresh,
            top_n=args.top,
            save_output=not args.no_save,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
