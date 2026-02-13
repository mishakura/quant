"""Abstract base class for cross-sectional factor implementations.

All factors implement :class:`Factor` so that the backtesting engine,
registry, and downstream portfolio construction code can treat them
uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Factor(ABC):
    """Base class for all factor implementations.

    Subclasses must implement :meth:`compute` (scoring logic) and
    :meth:`validate_data` (input checks).

    Attributes
    ----------
    name : str
        Human-readable factor name (e.g. ``"12M_Momentum_FIP"``).
    lookback : int
        Minimum number of trading days of history required.
    """

    def __init__(self, name: str, lookback: int) -> None:
        self.name = name
        self.lookback = lookback

    @abstractmethod
    def compute(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute factor scores for all assets.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data with ``DatetimeIndex`` rows and ticker columns.
            Must contain at least ``self.lookback`` rows.
        **kwargs : Any
            Factor-specific additional data.

        Returns
        -------
        pd.DataFrame
            Factor scores.  Must contain at least the columns
            ``['ticker', 'score', 'rank']``.  Additional metadata columns
            are allowed (e.g. ``'momentum'``, ``'fip'``, ``'skewness'``).
            Higher ``score`` always means *more desirable* for a long position.

        Raises
        ------
        InsufficientDataError
            If input data has fewer rows than ``self.lookback``.
        DataError
            If data validation fails.
        """
        ...

    @abstractmethod
    def validate_data(self, prices: pd.DataFrame, **kwargs: Any) -> None:
        """Validate that input data meets factor requirements.

        Raises on failure (does **not** return a boolean).

        Parameters
        ----------
        prices : pd.DataFrame
            Price data to validate.
        **kwargs : Any
            Additional data to validate.

        Raises
        ------
        InsufficientDataError
            If insufficient history.
        DataError
            If data quality checks fail.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, lookback={self.lookback})"


def rank_scores(scores: pd.Series) -> pd.Series:
    """Compute percentile ranks (0–100) for a score series.

    Higher raw scores receive higher ranks.

    Parameters
    ----------
    scores : pd.Series
        Numerical factor scores.

    Returns
    -------
    pd.Series
        Percentile ranks in [0, 100].
    """
    return scores.rank(pct=True) * 100.0
