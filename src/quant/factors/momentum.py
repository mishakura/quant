"""12-month price momentum factor with FIP refinement.

Methodology (Jegadeesh & Titman 1993, with FIP filter):

1. Resample daily prices to month-end.
2. Compute 12-month cumulative return **excluding** the most recent month
   (reversal avoidance).
3. Select top ``top_momentum_pct`` by momentum.
4. For those, compute FIP = ``pct_negative_days − pct_positive_days``
   over the last 252 daily returns.
5. Select top ``top_fip_pct`` by ascending FIP (lowest FIP is best).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant.exceptions import DataError, InsufficientDataError
from quant.factors.base import Factor, rank_scores
from quant.factors.registry import register_factor
from quant.utils.constants import TRADING_DAYS_PER_YEAR
from quant.utils.logging import get_logger

logger = get_logger(__name__)


@register_factor("momentum_12m")
class MomentumFactor(Factor):
    """12-month momentum with recent-month skip and FIP filtering.

    Parameters
    ----------
    lookback : int
        Trading days of history required (default 280 ≈ 13 months).
    top_momentum_pct : float
        Fraction of universe kept after momentum ranking (default 0.20).
    top_fip_pct : float
        Fraction of momentum winners kept after FIP ranking (default 0.50).
    """

    def __init__(
        self,
        lookback: int = 280,
        top_momentum_pct: float = 0.20,
        top_fip_pct: float = 0.50,
    ) -> None:
        super().__init__(name="12M_Momentum_FIP", lookback=lookback)
        self.top_momentum_pct = top_momentum_pct
        self.top_fip_pct = top_fip_pct

    def validate_data(self, prices: pd.DataFrame, **kwargs: Any) -> None:
        """Check minimum history length."""
        if len(prices) < self.lookback:
            raise InsufficientDataError(
                required=self.lookback, actual=len(prices),
                context=self.name,
            )
        if prices.isna().all().any():
            all_nan = list(prices.columns[prices.isna().all()])
            raise DataError(
                f"Tickers with all-NaN prices: {all_nan}"
            )

    def compute(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute momentum scores with FIP refinement.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily adjusted close prices (DatetimeIndex × tickers).

        Returns
        -------
        pd.DataFrame
            Columns ``['ticker', 'momentum', 'fip', 'score', 'rank']``.
            ``score`` = −FIP (higher is better).
        """
        self.validate_data(prices)

        # --- Step 1: monthly returns & 12-month momentum ---
        monthly_prices = prices.resample("ME").last()
        monthly_returns = monthly_prices.pct_change()

        results: list[dict[str, Any]] = []
        for ticker in prices.columns:
            mret = monthly_returns[ticker].dropna()
            if len(mret) < 12:
                logger.debug(
                    "Skipping %s: only %d monthly returns (need 12)", ticker, len(mret),
                )
                continue
            # Last 12 monthly returns, skip most recent month
            ret_12m = mret.iloc[-12:]
            momentum = float((ret_12m.iloc[:-1] + 1).prod() - 1)
            if np.isnan(momentum):
                continue
            results.append({"ticker": ticker, "momentum": momentum})

        if not results:
            raise DataError("No valid momentum scores computed for any ticker.")

        mom_df = pd.DataFrame(results).sort_values("momentum", ascending=False)

        # --- Step 2: top fraction by momentum ---
        top_n = max(1, int(len(mom_df) * self.top_momentum_pct))
        top_mom = mom_df.head(top_n).copy()

        # --- Step 3: FIP for top momentum tickers ---
        fip_values: list[float] = []
        for ticker in top_mom["ticker"]:
            daily_ret = prices[ticker].pct_change().dropna()
            daily_ret = daily_ret.iloc[-TRADING_DAYS_PER_YEAR:]
            num_days = len(daily_ret)
            if num_days > 0:
                pct_neg = (daily_ret < 0).sum() / num_days
                pct_pos = (daily_ret > 0).sum() / num_days
                fip_values.append(float(pct_neg - pct_pos))
            else:
                fip_values.append(np.nan)

        top_mom["fip"] = fip_values
        top_mom = top_mom.dropna(subset=["fip"])

        # --- Step 4: top fraction by FIP (ascending – lower is better) ---
        top_mom = top_mom.sort_values("fip", ascending=True)
        final_n = max(1, int(len(top_mom) * self.top_fip_pct))
        final = top_mom.head(final_n).copy()

        # Score = −FIP so that higher score = better
        final["score"] = -final["fip"]
        final["rank"] = rank_scores(final["score"]).values
        return final[["ticker", "momentum", "fip", "score", "rank"]].reset_index(drop=True)
