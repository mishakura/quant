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


def load_prices(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load cached close prices from the local Parquet store.

    Parameters
    ----------
    tickers : list[str] or None
        Ticker symbols to load. If None, loads all cached tickers.
    start : str or None
        Start date ``YYYY-MM-DD``. If None, loads from the earliest date.
    end : str or None
        End date ``YYYY-MM-DD``. If None, loads to the latest date.
    cache_path : str, Path, or None
        Override path to the Parquet cache file. Defaults to
        ``data/raw/prices.parquet``.

    Returns
    -------
    pd.DataFrame
        Close prices with DatetimeIndex and ticker columns.

    Raises
    ------
    DataError
        If the cache file does not exist.
    """
    from quant.config import RAW_DATA_DIR

    if cache_path is None:
        cache_path = RAW_DATA_DIR / "prices.parquet"
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise DataError(
            f"No cached price data found at {cache_path}. "
            "Run 'python scripts/update_data.py --tickers ...' to download data first."
        )

    df = pd.read_parquet(cache_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if tickers is not None:
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            logger.warning("Tickers not in cache: %s", missing)
        available = [t for t in tickers if t in df.columns]
        df = df[available]

    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]

    logger.info("Loaded %d rows x %d tickers from cache", len(df), len(df.columns))
    return df


def load_ticker_info(
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load cached ticker info (market cap, quote type, sector, etc.).

    Parameters
    ----------
    cache_path : str, Path, or None
        Override path to the Parquet cache file. Defaults to
        ``data/raw/ticker_info.parquet``.

    Returns
    -------
    pd.DataFrame
        Ticker info indexed by ticker symbol with columns:
        market_cap, quote_type, short_name, sector, industry.

    Raises
    ------
    DataError
        If the cache file does not exist.
    """
    from quant.config import RAW_DATA_DIR

    if cache_path is None:
        cache_path = RAW_DATA_DIR / "ticker_info.parquet"
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise DataError(
            f"No cached ticker info found at {cache_path}. "
            "Run 'python scripts/update_data.py --universe default --start 2015-01-01' to download data first."
        )

    df = pd.read_parquet(cache_path)
    logger.info("Loaded ticker info for %d tickers from cache", len(df))
    return df


def load_fundamentals(
    tickers: list[str] | None = None,
    statements: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load cached fundamental data from the local Parquet store.

    Parameters
    ----------
    tickers : list[str] or None
        Ticker symbols to load. If None, loads all cached tickers.
    statements : list[str] or None
        Which statements to load (e.g. ``['income', 'balance']``).
        If None, loads all four: income, balance, cashflow, income_ttm.
    cache_dir : str, Path, or None
        Override directory containing the fundamental Parquet files.
        Defaults to ``data/raw/``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of statement name to DataFrame with MultiIndex
        ``(ticker, field)`` rows and period-end date columns.

    Raises
    ------
    DataError
        If no fundamental cache files exist.
    """
    from quant.config import RAW_DATA_DIR

    if cache_dir is None:
        cache_dir = RAW_DATA_DIR
    cache_dir = Path(cache_dir)

    # Check that at least one cache file exists
    from quant.data.fundamentals import _STATEMENT_FILES

    any_exists = any((cache_dir / f).exists() for f in _STATEMENT_FILES.values())
    if not any_exists:
        raise DataError(
            f"No cached fundamental data found in {cache_dir}. "
            "Run 'python scripts/update_data.py --fundamentals' to download data first."
        )

    from quant.data.fundamentals import FundamentalStore

    store = FundamentalStore(cache_dir=cache_dir)
    result = store.load(tickers=tickers, statements=statements)
    total_tickers = max(
        (df.index.get_level_values(0).nunique() for df in result.values() if not df.empty),
        default=0,
    )
    logger.info("Loaded fundamentals for %d tickers from cache", total_tickers)
    return result


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
