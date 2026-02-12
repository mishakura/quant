"""Local data loaders — read cached data from disk.

These functions load data that has already been downloaded and saved locally
(CSV, Parquet, Excel). They do not fetch from remote sources.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from quant.exceptions import DataError
from quant.utils.logging import get_logger

logger = get_logger(__name__)


def load_price_csv(
    path: str | Path,
    date_column: str | int = 0,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load a price CSV file with a datetime index.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    date_column : str or int
        Column name or index to parse as dates.
    tickers : list[str] or None
        If provided, select only these columns after loading.

    Returns
    -------
    pd.DataFrame
        Price data with DatetimeIndex and ticker columns.

    Raises
    ------
    DataError
        If the file doesn't exist or contains no data.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"Price file not found: {path}")

    df = pd.read_csv(path, index_col=date_column, parse_dates=True)
    df = df.sort_index()

    if df.empty:
        raise DataError(f"Price file is empty: {path}")

    if tickers is not None:
        missing = set(tickers) - set(df.columns)
        if missing:
            logger.warning("Tickers not found in %s: %s", path.name, missing)
        available = [t for t in tickers if t in df.columns]
        df = df[available]

    logger.info("Loaded %d rows x %d columns from %s", len(df), len(df.columns), path.name)
    return df


def load_returns_from_prices(
    prices: pd.DataFrame,
    method: str = "arithmetic",
) -> pd.DataFrame:
    """Compute returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with DatetimeIndex.
    method : str
        ``'arithmetic'`` for simple returns ``(P_t / P_{t-1}) - 1``, or
        ``'log'`` for log returns ``ln(P_t / P_{t-1})``.

    Returns
    -------
    pd.DataFrame
        Return data (first row is dropped as NaN).

    Raises
    ------
    ValueError
        If *method* is not ``'arithmetic'`` or ``'log'``.
    """
    if method == "arithmetic":
        returns = prices.pct_change().iloc[1:]
    elif method == "log":
        returns = np.log(prices / prices.shift(1)).iloc[1:]
    else:
        raise ValueError(f"Unknown return method '{method}'. Use 'arithmetic' or 'log'.")

    return returns


def load_weights_excel(
    path: str | Path,
    sheet_name: str | int = 0,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Load portfolio weights from an Excel file.

    Parameters
    ----------
    path : str or Path
        Path to the Excel file.
    sheet_name : str or int
        Sheet to read.
    date_column : str or None
        Column name for dates. If provided, it becomes the index.
        If None, the first column is tried.

    Returns
    -------
    pd.DataFrame
        Weight data with DatetimeIndex (if date column found) and asset columns.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"Weights file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    # Find date column
    if date_column is None:
        for col in df.columns:
            if isinstance(col, str) and col.strip().lower() in ("fecha", "date"):
                date_column = col
                break

    if date_column is not None and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

    return df
