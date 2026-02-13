"""Low volatility anomaly factor.

Ranks assets by annualised volatility in ascending order — lower
volatility receives a higher score.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant.analytics.performance import annualized_volatility
from quant.exceptions import InsufficientDataError
from quant.factors.base import Factor, rank_scores
from quant.factors.registry import register_factor
from quant.utils.constants import TRADING_DAYS_PER_YEAR
from quant.utils.logging import get_logger

logger = get_logger(__name__)


@register_factor("low_volatility")
class LowVolatilityFactor(Factor):
    """Low volatility anomaly (lower vol = higher score).

    Parameters
    ----------
    lookback : int
        Trading days for volatility calculation (default 252).
    periods_per_year : int
        Annualisation factor (default 252).
    """

    def __init__(
        self,
        lookback: int = TRADING_DAYS_PER_YEAR,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(name="Low_Volatility", lookback=lookback)
        self.periods_per_year = periods_per_year

    def validate_data(self, prices: pd.DataFrame, **kwargs: Any) -> None:
        """Check minimum history length."""
        if len(prices) < self.lookback:
            raise InsufficientDataError(
                required=self.lookback, actual=len(prices),
                context=self.name,
            )

    def compute(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute annualised volatility scores.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily adjusted close prices (DatetimeIndex × tickers).

        Returns
        -------
        pd.DataFrame
            Columns ``['ticker', 'volatility', 'score', 'rank']``.
            ``score`` = −volatility (higher is better).
        """
        self.validate_data(prices)

        returns = prices.pct_change().dropna()

        vols: dict[str, float] = {}
        for ticker in prices.columns:
            ret = returns[ticker].dropna()
            if len(ret) >= self.lookback:
                vol = annualized_volatility(
                    ret.iloc[-self.lookback :], self.periods_per_year,
                )
                vols[ticker] = float(vol)
            else:
                logger.debug(
                    "Skipping %s: only %d returns (need %d)",
                    ticker, len(ret), self.lookback,
                )

        if not vols:
            raise InsufficientDataError(
                required=self.lookback, actual=0,
                context="No ticker had enough returns for volatility calculation",
            )

        result = pd.DataFrame({
            "ticker": list(vols.keys()),
            "volatility": list(vols.values()),
        })

        # Score = −volatility (lower vol = higher score)
        result["score"] = -result["volatility"]
        result["rank"] = rank_scores(result["score"]).values
        return result.sort_values("rank", ascending=False).reset_index(drop=True)
