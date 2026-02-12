"""Abstract base class for data providers.

All data source adapters (yfinance, BYMA, CME, etc.) implement this
interface so that downstream code is decoupled from data source specifics.
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Interface for market data providers.

    Subclasses must implement :meth:`fetch_prices` and :meth:`fetch_ohlcv`.
    """

    @abstractmethod
    def fetch_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for the given tickers.

        Parameters
        ----------
        tickers : list[str]
            Asset symbols to download.
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str or None
            End date in ``YYYY-MM-DD`` format. Defaults to today.
        auto_adjust : bool
            Whether to adjust for splits and dividends.

        Returns
        -------
        pd.DataFrame
            Close prices with DatetimeIndex rows and ticker columns,
            sorted by date ascending and columns alphabetically.
        """
        ...

    @abstractmethod
    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch full OHLCV data for the given tickers.

        Parameters
        ----------
        tickers : list[str]
            Asset symbols to download.
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str or None
            End date in ``YYYY-MM-DD`` format. Defaults to today.
        auto_adjust : bool
            Whether to adjust for splits and dividends.

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with (Date, Ticker) or multi-level columns
            containing Open, High, Low, Close, Volume.
        """
        ...
