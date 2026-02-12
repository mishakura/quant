"""Data validation helpers for returns, weights, and prices.

These functions raise ``DataError`` or ``InsufficientDataError`` from
``quant.exceptions`` so callers get clear, actionable error messages.
"""

import numpy as np
import pandas as pd

from quant.exceptions import DataError, InsufficientDataError


def validate_returns(
    returns: pd.Series | pd.DataFrame,
    min_observations: int = 2,
    allow_nan: bool = False,
    context: str = "",
) -> pd.Series | pd.DataFrame:
    """Validate a return series or DataFrame.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Return data to validate.
    min_observations : int
        Minimum number of non-NaN observations required.
    allow_nan : bool
        If False, raise on any NaN values.
    context : str
        Optional context string for error messages.

    Returns
    -------
    pd.Series or pd.DataFrame
        The validated input (unchanged).

    Raises
    ------
    DataError
        If input is not a Series/DataFrame or is empty.
    InsufficientDataError
        If there are fewer than *min_observations* non-NaN values.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise DataError(f"Expected pd.Series or pd.DataFrame, got {type(returns).__name__}")

    if returns.empty:
        raise DataError(f"Return data is empty. {context}".strip())

    if isinstance(returns, pd.Series):
        n_valid = int(returns.notna().sum())
    else:
        n_valid = int(returns.notna().all(axis=1).sum())

    if n_valid < min_observations:
        raise InsufficientDataError(
            required=min_observations, actual=n_valid, context=context
        )

    if not allow_nan:
        if isinstance(returns, pd.DataFrame):
            has_nan = returns.isna().any().any()
            nan_count = int(returns.isna().sum().sum())
        else:
            has_nan = bool(returns.isna().any())
            nan_count = int(returns.isna().sum())
        if has_nan:
            raise DataError(
                f"Return data contains {nan_count} NaN values. "
                f"Use allow_nan=True or clean data first. {context}".strip()
            )

    return returns


def validate_weights(
    weights: pd.Series,
    tolerance: float = 1e-6,
    allow_negative: bool = False,
    context: str = "",
) -> pd.Series:
    """Validate portfolio weights.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weight vector.
    tolerance : float
        Tolerance for sum-to-one check.
    allow_negative : bool
        If False, raise on negative weights (long-only constraint).
    context : str
        Optional context string for error messages.

    Returns
    -------
    pd.Series
        The validated input (unchanged).

    Raises
    ------
    DataError
        If weights don't sum to ~1 or contain invalid values.
    """
    if not isinstance(weights, pd.Series):
        raise DataError(f"Expected pd.Series, got {type(weights).__name__}")

    if weights.empty:
        raise DataError(f"Weight vector is empty. {context}".strip())

    if weights.isna().any():
        raise DataError(f"Weight vector contains NaN values. {context}".strip())

    weight_sum = weights.sum()
    if abs(weight_sum - 1.0) > tolerance:
        raise DataError(
            f"Weights sum to {weight_sum:.6f}, expected 1.0 "
            f"(tolerance={tolerance}). {context}".strip()
        )

    if not allow_negative and (weights < -tolerance).any():
        neg_assets = weights[weights < -tolerance].index.tolist()
        raise DataError(
            f"Negative weights found for {neg_assets}. "
            f"Set allow_negative=True for long-short portfolios. {context}".strip()
        )

    return weights


def check_no_negative_prices(prices: pd.Series | pd.DataFrame, context: str = "") -> None:
    """Raise DataError if any price values are negative.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price data to validate.
    context : str
        Optional context string for error messages.

    Raises
    ------
    DataError
        If any price value is negative.
    """
    if isinstance(prices, pd.Series):
        neg_mask = prices < 0
        if neg_mask.any():
            n_neg = int(neg_mask.sum())
            first_date = prices.index[neg_mask][0]
            raise DataError(
                f"Found {n_neg} negative price(s). "
                f"First occurrence at {first_date}. {context}".strip()
            )
    elif isinstance(prices, pd.DataFrame):
        neg_mask = prices < 0
        if neg_mask.any().any():
            total_neg = int(neg_mask.sum().sum())
            # Find first occurrence
            for col in prices.columns:
                col_neg = neg_mask[col]
                if col_neg.any():
                    first_date = prices.index[col_neg][0]
                    raise DataError(
                        f"Found {total_neg} negative price(s). "
                        f"First occurrence: {col} at {first_date}. {context}".strip()
                    )
