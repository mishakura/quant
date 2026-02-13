"""Sector momentum rotation using SPDR Select Sector ETFs.

Ranks the 11 SPDR sector ETFs by N-month simple return and selects
the top *K* sectors for an equal-weight allocation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant.exceptions import DataError, InsufficientDataError
from quant.factors.base import Factor, rank_scores
from quant.factors.registry import register_factor
from quant.utils.logging import get_logger

logger = get_logger(__name__)

SPDR_SECTOR_ETFS: list[str] = [
    "XLB",   # Materials
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Technology
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
    "XLC",   # Communication Services
    "XLRE",  # Real Estate
]


@register_factor("sector_momentum_11m")
class SectorMomentumFactor(Factor):
    """Sector momentum rotation (top N by simple return).

    Parameters
    ----------
    lookback : int
        Trading days for the return window (default 231 ≈ 11 months).
    top_n : int
        Number of sectors to select (default 3).
    """

    def __init__(
        self,
        lookback: int = 231,
        top_n: int = 3,
    ) -> None:
        super().__init__(name="Sector_Momentum_11M", lookback=lookback)
        self.top_n = top_n

    def validate_data(self, prices: pd.DataFrame, **kwargs: Any) -> None:
        """Check minimum history and warn about missing ETFs."""
        if len(prices) < self.lookback:
            raise InsufficientDataError(
                required=self.lookback, actual=len(prices),
                context=self.name,
            )
        missing = set(SPDR_SECTOR_ETFS) - set(prices.columns)
        if missing:
            logger.warning("Missing SPDR ETFs in price data: %s", sorted(missing))

    def compute(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute sector momentum scores.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily adjusted close prices for sector ETFs.

        Returns
        -------
        pd.DataFrame
            Top ``self.top_n`` sectors.  Columns:
            ``['ticker', 'return', 'score', 'rank']``.
        """
        self.validate_data(prices)

        available = [etf for etf in SPDR_SECTOR_ETFS if etf in prices.columns]
        if not available:
            raise DataError("No SPDR sector ETFs found in price data columns.")

        returns: dict[str, float] = {}
        for etf in available:
            p = prices[etf].dropna()
            if len(p) >= self.lookback:
                first = p.iloc[-self.lookback]
                last = p.iloc[-1]
                if first > 0:
                    returns[etf] = float(last / first - 1)

        if not returns:
            raise DataError("No sector ETF had sufficient price history.")

        result = pd.DataFrame({
            "ticker": list(returns.keys()),
            "return": list(returns.values()),
        })

        result["score"] = result["return"]
        result["rank"] = rank_scores(result["score"]).values
        result = result.sort_values("rank", ascending=False).reset_index(drop=True)

        return result.head(self.top_n)
