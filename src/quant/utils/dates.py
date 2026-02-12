"""Date utilities for trading calendars, rebalancing, and frequency inference.

Extracted from performance/performance.py, strategies/hedging/dynamic_hedge/dhedging.py,
and strategies/factor/momentum/momentum.py.
"""

import pandas as pd

from quant.utils.constants import (
    MONTHS_PER_YEAR,
    QUARTER_END_DATES,
    QUARTERS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    WEEKS_PER_YEAR,
)


def is_quarter_end(date: pd.Timestamp | str) -> bool:
    """Check whether a date falls on a calendar quarter-end.

    Parameters
    ----------
    date : pd.Timestamp or str
        Date to check. Strings are parsed via ``pd.to_datetime``.

    Returns
    -------
    bool
        True if date is March 31, June 30, September 30, or December 31.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return (date.month, date.day) in QUARTER_END_DATES


def last_quarter_end(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the most recent quarter-end date present in *dates*.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Sorted or unsorted index of dates to search.

    Returns
    -------
    pd.Timestamp
        The latest date in *dates* that falls on a quarter-end.

    Raises
    ------
    ValueError
        If no quarter-end date is found in *dates*.
    """
    quarter_ends = [d for d in dates if is_quarter_end(d)]
    if not quarter_ends:
        raise ValueError("No quarter-end dates found in the provided index.")
    return max(quarter_ends)


def infer_periods_per_year(dates: pd.DatetimeIndex) -> int:
    """Infer the annualization factor from the median gap between dates.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Datetime index whose frequency should be inferred.

    Returns
    -------
    int
        Estimated number of periods per year:
        252 (daily), 52 (weekly), 12 (monthly), 4 (quarterly).
        Defaults to 252 if the index has fewer than 2 dates.
    """
    if len(dates) < 2:
        return TRADING_DAYS_PER_YEAR

    diffs = dates.to_series().diff().dt.days.dropna()
    median_days = diffs.median()

    if median_days <= 2:
        return TRADING_DAYS_PER_YEAR
    if median_days <= 9:
        return WEEKS_PER_YEAR
    if median_days <= 40:
        return MONTHS_PER_YEAR
    if median_days <= 120:
        return QUARTERS_PER_YEAR
    return TRADING_DAYS_PER_YEAR


def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "quarterly",
    months: list[int] | None = None,
) -> pd.DatetimeIndex:
    """Select rebalance dates from a trading calendar.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full set of available trading dates (sorted ascending).
    frequency : str
        One of ``'monthly'``, ``'quarterly'``, ``'annual'``.
        Ignored if *months* is provided.
    months : list[int] | None
        Explicit list of months in which to rebalance (e.g. ``[2, 5, 8, 11]``).
        When provided, *frequency* is ignored.

    Returns
    -------
    pd.DatetimeIndex
        Subset of *dates* — the last trading day of each rebalance period.
    """
    dates = dates.sort_values()
    series = dates.to_series()

    if months is not None:
        mask = series.dt.month.isin(months)
        filtered = series[mask]
        # Last trading day per (year, month)
        result = filtered.groupby([filtered.dt.year, filtered.dt.month]).last()
        return pd.DatetimeIndex(result.values)

    freq_map = {
        "monthly": "M",
        "quarterly": "Q",
        "annual": "Y",
    }
    if frequency not in freq_map:
        raise ValueError(
            f"Unknown frequency '{frequency}'. Choose from {list(freq_map.keys())}."
        )

    # Group by period and take the last trading day in each period
    period_labels = series.dt.to_period(freq_map[frequency])
    result = series.groupby(period_labels).last()
    return pd.DatetimeIndex(result.values)
