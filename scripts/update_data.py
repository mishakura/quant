#!/usr/bin/env python
"""Download or refresh market price data into the local Parquet cache.

Usage
-----
# Download full universe from 2015
python scripts/update_data.py --universe default --start 2015-01-01

# Update all cached tickers to today
python scripts/update_data.py

# Download specific tickers
python scripts/update_data.py --tickers AAPL MSFT SPY --start 2015-01-01
"""

import argparse
import sys
from pathlib import Path

import yaml

from quant.config import CONFIGS_DIR, ensure_directories
from quant.data.providers.yfinance import YFinanceProvider
from quant.data.store import PriceStore
from quant.utils.logging import get_logger

logger = get_logger(__name__)


def load_universe(name: str) -> list[str]:
    """Load ticker list from a universe YAML file.

    Parameters
    ----------
    name : str
        Universe name (e.g. ``'default'``). Looks for
        ``configs/universes/{name}.yaml``.

    Returns
    -------
    list[str]
        Deduplicated, sorted list of ticker symbols.
    """
    path = CONFIGS_DIR / "universes" / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in (CONFIGS_DIR / "universes").glob("*.yaml")]
        print(f"Universe '{name}' not found at {path}")
        print(f"Available universes: {available}")
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    tickers: list[str] = []
    for key, value in config.items():
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            tickers.extend(value)

    unique = sorted(set(tickers))
    logger.info("Loaded universe '%s': %d tickers from %s", name, len(unique), path)
    return unique


def main() -> None:
    """Parse CLI args and run the price update."""
    parser = argparse.ArgumentParser(
        description="Download or refresh market price data into the local Parquet cache.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker symbols to download (e.g. AAPL MSFT SPY).",
    )
    group.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Universe name to load from configs/universes/<name>.yaml (e.g. 'default').",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD for new ticker downloads. "
        "Defaults to the earliest date in the existing cache, or 2000-01-01.",
    )
    args = parser.parse_args()

    ensure_directories()

    # Resolve ticker list
    tickers = args.tickers
    if args.universe:
        tickers = load_universe(args.universe)
        print(f"Universe '{args.universe}': {len(tickers)} tickers")

    provider = YFinanceProvider(progress=True)
    store = PriceStore(provider)

    cached = store.list_cached()
    if tickers is None and not cached:
        logger.error("No cached tickers and no --tickers/--universe specified. Nothing to do.")
        print("No cached tickers found. Provide tickers, e.g.:")
        print("  python scripts/update_data.py --universe default --start 2015-01-01")
        print("  python scripts/update_data.py --tickers SPY AAPL MSFT --start 2020-01-01")
        sys.exit(1)

    result = store.update(tickers=tickers, start=args.start)

    print(f"\nCache updated: {len(result.columns)} tickers, {len(result)} trading days")
    if not result.empty:
        print(f"  Date range: {result.index[0].date()} to {result.index[-1].date()}")
    print(f"  Cache file: {store.cache_path}")


if __name__ == "__main__":
    main()
