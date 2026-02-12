"""Data cleaning and alignment utilities.

Functions for aligning multi-asset data to common date ranges,
handling missing values, and validating data quality before analysis.
"""

import pandas as pd

from quant.exceptions import DataError
from quant.utils.logging import get_logger

logger = get_logger(__name__)


def find_common_start_date(data: pd.DataFrame) -> pd.Timestamp:
    """Find the earliest date where all columns have data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with DatetimeIndex and asset columns.

    Returns
    -------
    pd.Timestamp
        The latest first-valid-index across all columns.

    Raises
    ------
    DataError
        If any column is entirely NaN.
    """
    first_dates = {}
    for col in data.columns:
        first = data[col].first_valid_index()
        if first is None:
            raise DataError(f"Column '{col}' has no valid data.")
        first_dates[col] = first

    common_start = max(first_dates.values())
    logger.info(
        "Common start date: %s (latest of %d assets)",
        common_start.strftime("%Y-%m-%d"),
        len(data.columns),
    )
    return pd.Timestamp(common_start)


def trim_to_common_history(data: pd.DataFrame) -> pd.DataFrame:
    """Trim data to the period where all assets have overlapping data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with DatetimeIndex and asset columns. May contain NaNs
        at the beginning of columns with shorter histories.

    Returns
    -------
    pd.DataFrame
        Trimmed DataFrame starting from the common start date,
        with any remaining NaN rows dropped.
    """
    common_start = find_common_start_date(data)
    trimmed = data.loc[common_start:].copy()

    n_before = len(trimmed)
    trimmed = trimmed.dropna()
    n_dropped = n_before - len(trimmed)

    if n_dropped > 0:
        logger.warning("Dropped %d rows with NaN after alignment.", n_dropped)

    return trimmed


def align_to_common_dates(
    *dataframes: pd.DataFrame | pd.Series,
    join: str = "inner",
) -> list[pd.DataFrame | pd.Series]:
    """Align multiple DataFrames/Series to their common date index.

    Parameters
    ----------
    *dataframes : pd.DataFrame or pd.Series
        Two or more objects to align.
    join : str
        Join method: ``'inner'`` (default) keeps only overlapping dates,
        ``'outer'`` keeps all dates (filling gaps with NaN).

    Returns
    -------
    list[pd.DataFrame or pd.Series]
        Aligned objects with the same DatetimeIndex.

    Raises
    ------
    ValueError
        If fewer than 2 objects are provided.
    """
    if len(dataframes) < 2:
        raise ValueError("Need at least 2 objects to align.")

    aligned = pd.concat(
        [df for df in dataframes],
        axis=1,
        join=join,
    )

    results = []
    col_offset = 0
    for df in dataframes:
        if isinstance(df, pd.Series):
            results.append(aligned.iloc[:, col_offset])
            col_offset += 1
        else:
            n_cols = len(df.columns)
            results.append(aligned.iloc[:, col_offset : col_offset + n_cols])
            col_offset += n_cols

    return results


def validate_price_data(prices: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Run standard quality checks on price data.

    Checks for:
    - Negative prices
    - Duplicate index entries
    - Non-monotonic index (dates out of order)

    Parameters
    ----------
    prices : pd.DataFrame
        Price data to validate.
    context : str
        Optional context for error messages.

    Returns
    -------
    pd.DataFrame
        The validated input (unchanged).

    Raises
    ------
    DataError
        If any quality check fails.
    """
    # Check for negative prices
    neg_mask = prices < 0
    if neg_mask.any().any():
        total = int(neg_mask.sum().sum())
        raise DataError(
            f"Found {total} negative price(s) in data. {context}".strip()
        )

    # Check for duplicate dates
    if prices.index.duplicated().any():
        n_dup = int(prices.index.duplicated().sum())
        raise DataError(
            f"Found {n_dup} duplicate date(s) in index. {context}".strip()
        )

    # Check index is sorted
    if not prices.index.is_monotonic_increasing:
        logger.warning("Price index is not sorted. Sorting automatically. %s", context)
        prices = prices.sort_index()

    return prices
