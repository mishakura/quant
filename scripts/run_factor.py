#!/usr/bin/env python
"""Run a factor and display the ranking.

Usage
-----
# List available factors
python scripts/run_factor.py --list

# Run momentum factor (top 20 by default)
python scripts/run_factor.py momentum_12m

# Run with options
python scripts/run_factor.py low_volatility --start 2018-01-01 --top 30

# Show full ranking
python scripts/run_factor.py skewness_252d --all
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yaml

# Ensure factor modules are imported so they register themselves
import quant.factors.low_volatility  # noqa: F401
import quant.factors.momentum  # noqa: F401
import quant.factors.sector_rotation  # noqa: F401
import quant.factors.skewness  # noqa: F401
import quant.factors.value  # noqa: F401
from quant.config import CONFIGS_DIR
from quant.data.loaders import load_fundamentals, load_prices, load_ticker_info
from quant.factors.registry import get_factor, list_factors
from quant.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Parse args and run the selected factor."""
    parser = argparse.ArgumentParser(
        description="Run a factor on cached price data and display the ranking.",
    )
    parser.add_argument(
        "factor",
        nargs="?",
        default=None,
        help="Factor name (e.g. momentum_12m, low_volatility, skewness_252d).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available factors and exit.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for price data (default: 2015-01-01).",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="default",
        help="Universe name to filter tickers (default: 'default'). Use 'all' to skip filtering.",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=None,
        help="Minimum market cap in USD (e.g. 2e9 for $2B). ETFs are always kept.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the top N results (by rank descending = best first).",
    )
    args = parser.parse_args()

    # List mode
    if args.list:
        factors = list_factors()
        print(f"Available factors ({len(factors)}):")
        for name in factors:
            print(f"  - {name}")
        return

    if args.factor is None:
        parser.print_help()
        sys.exit(1)

    # Load data
    print(f"Loading prices from {args.start}...")
    prices = load_prices(start=args.start)

    # Filter to universe tickers
    if args.universe != "all":
        universe_path = CONFIGS_DIR / "universes" / f"{args.universe}.yaml"
        if not universe_path.exists():
            print(f"Universe '{args.universe}' not found at {universe_path}")
            sys.exit(1)
        with open(universe_path) as f:
            config = yaml.safe_load(f)
        universe_tickers: list[str] = []
        for value in config.values():
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                universe_tickers.extend(value)
        universe_set = set(universe_tickers)
        available = [t for t in prices.columns if t in universe_set]
        prices = prices[available]
        print(f"  Filtered to universe '{args.universe}': {len(available)} tickers")

    print(f"  {prices.shape[0]} days x {prices.shape[1]} tickers\n")

    # Run factor (all factors expect prices, not returns)
    factor_cls = get_factor(args.factor)
    factor = factor_cls()
    print(f"Running factor: {args.factor} ({factor.__class__.__name__})...")

    # Auto-detect if factor needs fundamental data
    factor_kwargs: dict = {}
    if getattr(factor, "requires_fundamentals", False):
        print("  Loading fundamental data...")
        try:
            factor_kwargs["fundamentals"] = load_fundamentals()
        except Exception as e:
            print(f"  Error loading fundamentals: {e}")
            print("  Run 'python scripts/update_data.py --fundamentals' to download data first.")
            sys.exit(1)
        try:
            factor_kwargs["ticker_info"] = load_ticker_info()
        except Exception as e:
            print(f"  Error loading ticker info: {e}")
            print("  Run 'python scripts/update_data.py' to download ticker info first.")
            sys.exit(1)

    result = factor.compute(prices, **factor_kwargs)

    # Market cap filtering
    if args.min_market_cap is not None:
        min_cap = args.min_market_cap
        print(f"\nFiltering by market cap >= ${min_cap/1e9:.1f}B (ETFs always kept)...")
        try:
            info = load_ticker_info()
        except Exception:
            print("  No cached ticker info found. Run 'python scripts/update_data.py' first.")
            sys.exit(1)

        tickers_in_result = result["ticker"].tolist()
        keep = []
        excluded = []
        for ticker in tickers_in_result:
            if ticker not in info.index:
                excluded.append((ticker, "no info"))
                continue
            row = info.loc[ticker]
            quote_type = str(row.get("quote_type", "")).lower()
            if quote_type == "etf":
                keep.append(ticker)
                continue
            market_cap = row.get("market_cap", np.nan)
            if pd.notna(market_cap) and market_cap >= min_cap:
                keep.append(ticker)
            else:
                cap_str = f"${market_cap/1e9:.1f}B" if pd.notna(market_cap) else "unknown"
                excluded.append((ticker, cap_str))

        result = result[result["ticker"].isin(keep)].reset_index(drop=True)
        print(f"  Kept {len(keep)} tickers, excluded {len(excluded)}")
        if excluded:
            exc_str = ", ".join(f"{t} ({c})" for t, c in excluded[:10])
            if len(excluded) > 10:
                exc_str += f" ... and {len(excluded) - 10} more"
            print(f"  Excluded: {exc_str}")

    # Display
    result = result.sort_values("rank", ascending=False)

    if args.top is not None:
        result = result.head(args.top)

    pd.set_option("display.max_rows", None)
    print(f"\n{'=' * 60}")
    print(f"  {args.factor} — {len(result)} tickers")
    print(f"{'=' * 60}")
    print(result.to_string(index=False))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
