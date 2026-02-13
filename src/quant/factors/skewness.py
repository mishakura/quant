"""Return skewness factor following Asness et al. (2013).

Assets with more negative return skewness are preferred (lottery-ticket
avoidance).  The factor computes 252-day rolling skewness and provides
rank-weighted long/short portfolio weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from quant.exceptions import DataError, InsufficientDataError
from quant.factors.base import Factor, rank_scores
from quant.factors.registry import register_factor
from quant.utils.constants import TRADING_DAYS_PER_YEAR
from quant.utils.logging import get_logger

logger = get_logger(__name__)


@register_factor("skewness_252d")
class SkewnessFactor(Factor):
    """Return skewness factor (long negative skew, short positive skew).

    Parameters
    ----------
    lookback : int
        Rolling window in trading days (default 252).
    min_periods_pct : float
        Minimum data requirement as fraction of ``lookback`` (default 0.8).
    """

    def __init__(
        self,
        lookback: int = TRADING_DAYS_PER_YEAR,
        min_periods_pct: float = 0.8,
    ) -> None:
        super().__init__(name="Return_Skewness_252D", lookback=lookback)
        self.min_periods = int(lookback * min_periods_pct)

    def validate_data(self, prices: pd.DataFrame, **kwargs: Any) -> None:
        """Check minimum history length."""
        if len(prices) < self.lookback:
            raise InsufficientDataError(
                required=self.lookback, actual=len(prices),
                context=self.name,
            )

    def compute(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute rolling skewness scores.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily adjusted close prices (DatetimeIndex × tickers).

        Returns
        -------
        pd.DataFrame
            Columns ``['ticker', 'skewness', 'score', 'rank']``.
            ``score`` = −skewness (higher is better = more negative skew).
        """
        self.validate_data(prices)

        returns = prices.pct_change()

        # Rolling skewness per ticker
        skewness_data: dict[str, pd.Series] = {}
        for ticker in prices.columns:
            rolling_skew = returns[ticker].rolling(
                window=self.lookback,
                min_periods=self.min_periods,
            ).apply(lambda x: stats.skew(x, nan_policy="omit"), raw=True)
            skewness_data[ticker] = rolling_skew

        skewness_df = pd.DataFrame(skewness_data, index=prices.index)

        # Last available row with any valid data
        valid_rows = skewness_df.dropna(how="all")
        if valid_rows.empty:
            raise DataError("No valid skewness values computed for any ticker.")

        last_skew = valid_rows.iloc[-1].dropna()
        if last_skew.empty:
            raise DataError("No valid skewness values on last available date.")

        result = pd.DataFrame({
            "ticker": last_skew.index,
            "skewness": last_skew.values,
        })

        # Score = −skewness (more negative skew → higher score)
        result["score"] = -result["skewness"]
        result["rank"] = rank_scores(result["score"]).values
        return result.sort_values("rank", ascending=False).reset_index(drop=True)

    @staticmethod
    def compute_rank_weights(scores: pd.DataFrame) -> pd.Series:
        """Compute rank-weighted long/short portfolio.

        Follows Asness et al. (2013) / Koijen et al. (2018):

        * Rank assets by −skewness (more negative = higher rank).
        * De-mean: ``weight_i = rank_i − (N + 1) / 2``.
        * Scale positive weights to sum to +1, negative to sum to −1.

        Parameters
        ----------
        scores : pd.DataFrame
            Output of :meth:`compute` (must contain ``'ticker'`` and
            ``'skewness'`` columns).

        Returns
        -------
        pd.Series
            Portfolio weights indexed by ticker.
        """
        skew_values = pd.Series(
            scores["skewness"].values, index=scores["ticker"].values,
        )
        n = len(skew_values.dropna())
        if n <= 1:
            return pd.Series(0.0, index=skew_values.index)

        ranks = (-1 * skew_values).rank()
        demeaned = ranks - (n + 1) / 2

        weights = demeaned.copy()
        pos = weights[weights > 0]
        neg = weights[weights < 0]

        if len(pos) > 0:
            weights[weights > 0] = weights[weights > 0] / pos.sum()
        if len(neg) > 0:
            weights[weights < 0] = weights[weights < 0] / neg.sum() * -1

        return weights
