"""YFinance data provider — wraps yfinance for standardized market data access.

Replaces the 12+ scattered ``yf.download()`` calls across the codebase with
a single, consistent interface that handles edge cases (single ticker,
missing data, type normalization).
"""

import datetime

import pandas as pd
import yfinance as yf

from quant.data.providers.base import DataProvider
from quant.exceptions import DataError
from quant.utils.logging import get_logger

logger = get_logger(__name__)


class YFinanceProvider(DataProvider):
    """Data provider backed by the Yahoo Finance API via ``yfinance``.

    Parameters
    ----------
    progress : bool
        Show yfinance download progress bar (default: False).
    """

    def __init__(self, progress: bool = False) -> None:
        self.progress = progress

    def fetch_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices from Yahoo Finance.

        Parameters
        ----------
        tickers : list[str]
            Asset symbols (e.g. ``["SPY", "QQQ", "GLD"]``).
        start : str
            Start date ``YYYY-MM-DD``.
        end : str or None
            End date ``YYYY-MM-DD``. Defaults to today.
        auto_adjust : bool
            Adjust for splits/dividends.

        Returns
        -------
        pd.DataFrame
            Close prices, DatetimeIndex rows, ticker columns (sorted).

        Raises
        ------
        DataError
            If no data is returned for any ticker.
        """
        if end is None:
            end = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")

        logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)

        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=self.progress,
        )

        if data.empty:
            raise DataError(f"No data returned for tickers: {tickers}")

        # Extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            # Single ticker returns flat columns
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})

        # Ensure DataFrame (single ticker may return Series)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        # Sort columns alphabetically, sort index by date
        prices = prices.reindex(columns=sorted(prices.columns))
        prices = prices.sort_index()

        logger.info(
            "Downloaded %d rows x %d assets (%s to %s)",
            len(prices),
            len(prices.columns),
            prices.index[0].strftime("%Y-%m-%d"),
            prices.index[-1].strftime("%Y-%m-%d"),
        )

        return prices

    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch full OHLCV data from Yahoo Finance.

        Parameters
        ----------
        tickers : list[str]
            Asset symbols.
        start : str
            Start date ``YYYY-MM-DD``.
        end : str or None
            End date ``YYYY-MM-DD``. Defaults to today.
        auto_adjust : bool
            Adjust for splits/dividends.

        Returns
        -------
        pd.DataFrame
            OHLCV data with MultiIndex columns ``(field, ticker)``
            for multi-ticker requests, or flat columns for single ticker.

        Raises
        ------
        DataError
            If no data is returned.
        """
        if end is None:
            end = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")

        logger.info("Downloading OHLCV for %d tickers from %s to %s", len(tickers), start, end)

        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=self.progress,
        )

        if data.empty:
            raise DataError(f"No OHLCV data returned for tickers: {tickers}")

        return data.sort_index()
