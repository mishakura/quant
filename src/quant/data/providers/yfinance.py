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
        batch_size: int = 1,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices from Yahoo Finance.

        Downloads one ticker at a time to avoid yfinance silent failures
        when requesting many tickers in a single call.

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
        batch_size : int
            Number of tickers per yfinance call (default: 1).

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

        all_prices: list[pd.DataFrame] = []
        failed: list[str] = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            logger.debug("Batch %d/%d: %s", i // batch_size + 1, -(-len(tickers) // batch_size), batch)

            try:
                data = yf.download(
                    batch if len(batch) > 1 else batch[0],
                    start=start,
                    end=end,
                    auto_adjust=auto_adjust,
                    progress=self.progress,
                )

                if data.empty:
                    logger.warning("No data for: %s", batch)
                    failed.extend(batch)
                    continue

                # yfinance always returns MultiIndex columns (Price, Ticker)
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data["Close"]
                else:
                    prices = data[["Close"]].rename(columns={"Close": batch[0]})

                # Ensure DataFrame
                if isinstance(prices, pd.Series):
                    prices = prices.to_frame(name=batch[0])

                all_prices.append(prices)

            except Exception as e:
                logger.warning("Failed to download %s: %s", batch, e)
                failed.extend(batch)

        if not all_prices:
            raise DataError(f"No data returned for any tickers: {tickers}")

        if failed:
            logger.warning("Failed tickers (%d): %s", len(failed), failed)

        # Combine all batches
        combined = pd.concat(all_prices, axis=1)
        combined = combined.reindex(columns=sorted(combined.columns))
        combined = combined.sort_index()

        logger.info(
            "Downloaded %d rows x %d assets (%s to %s). %d failed.",
            len(combined),
            len(combined.columns),
            combined.index[0].strftime("%Y-%m-%d"),
            combined.index[-1].strftime("%Y-%m-%d"),
            len(failed),
        )

        return combined

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
