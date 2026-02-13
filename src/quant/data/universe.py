"""Asset universe definitions and filtering utilities.

Provides the :class:`Universe` container for managing ticker lists with
optional metadata filters (exchange, market cap).  Also contains predefined
universes extracted from the existing factor scripts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant.utils.logging import get_logger

logger = get_logger(__name__)

# US exchanges recognised by yfinance ``info['exchange']``
US_EXCHANGES: set[str] = {
    "NMS", "NYQ", "AMEX", "BATS", "ARCX", "PCX",
    "NGM", "NSC", "XASE", "XNAS", "XNYS",
}

# SPDR Select Sector ETFs
SPDR_SECTOR_ETFS: list[str] = [
    "XLB", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE",
]


class Universe:
    """Asset universe with optional filter criteria.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``'US_LARGE_CAP'``).
    tickers : list[str]
        Asset symbols.
    filters : dict[str, Any] | None
        Optional filter specs.  Recognised keys:

        * ``'exchanges'`` – ``list[str]`` of allowed exchange codes.
        * ``'min_market_cap'`` – ``float`` minimum market capitalisation (USD).
        * ``'force_include'`` – ``set[str]`` tickers that bypass exchange filter.
    """

    def __init__(
        self,
        name: str,
        tickers: list[str],
        filters: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.tickers = list(tickers)
        self.filters = filters or {}

    def __repr__(self) -> str:
        return f"Universe(name={self.name!r}, n_tickers={len(self.tickers)})"

    def __len__(self) -> int:
        return len(self.tickers)


def filter_by_exchange(
    tickers: list[str],
    allowed_exchanges: set[str] | list[str],
    force_include: set[str] | None = None,
    verbose: bool = False,
) -> list[str]:
    """Filter tickers by exchange using yfinance metadata.

    Parameters
    ----------
    tickers : list[str]
        Tickers to filter.
    allowed_exchanges : set[str] | list[str]
        Exchange codes to keep (e.g. ``US_EXCHANGES``).
    force_include : set[str] | None
        Tickers that always pass the filter regardless of exchange.
    verbose : bool
        Log skipped tickers.

    Returns
    -------
    list[str]
        Filtered ticker list.
    """
    import yfinance as yf

    allowed = {ex.upper() for ex in allowed_exchanges}
    force = force_include or set()
    filtered: list[str] = []

    for ticker in tickers:
        if ticker in force:
            filtered.append(ticker)
            continue
        try:
            info = yf.Ticker(ticker).info
            exchange = info.get("exchange", "").upper()
            if exchange in allowed:
                filtered.append(ticker)
            elif verbose:
                logger.info("Skipping %s: exchange %s not in allowed set", ticker, exchange)
        except Exception as exc:
            if verbose:
                logger.warning("Skipping %s: error fetching info – %s", ticker, exc)

    return filtered


def filter_by_market_cap(
    tickers: list[str],
    min_market_cap: float,
    verbose: bool = False,
) -> list[str]:
    """Filter tickers by minimum market capitalisation.

    Parameters
    ----------
    tickers : list[str]
        Tickers to filter.
    min_market_cap : float
        Minimum market cap in USD.
    verbose : bool
        Log skipped tickers.

    Returns
    -------
    list[str]
        Filtered ticker list.
    """
    import yfinance as yf

    filtered: list[str] = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            market_cap = info.get("marketCap", 0) or 0
            if market_cap >= min_market_cap:
                filtered.append(ticker)
            elif verbose:
                logger.info(
                    "Skipping %s: market cap $%,.0f < $%,.0f",
                    ticker, market_cap, min_market_cap,
                )
        except Exception as exc:
            if verbose:
                logger.warning("Skipping %s: error fetching info – %s", ticker, exc)

    return filtered
