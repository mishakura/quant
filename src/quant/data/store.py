"""Local Parquet price store — download once, read from disk thereafter.

Wraps a DataProvider to cache prices as a single Parquet file. Subsequent
calls read from the cache and only download missing tickers or new dates.
"""

import datetime
from pathlib import Path

import pandas as pd

from quant.config import RAW_DATA_DIR
from quant.data.providers.base import DataProvider
from quant.exceptions import DataError
from quant.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_CACHE_FILE = "prices.parquet"


class PriceStore:
    """Persistent Parquet cache for close prices.

    Parameters
    ----------
    provider : DataProvider
        Data source used to download missing data.
    cache_dir : Path
        Directory where the Parquet cache file is stored.
        Defaults to ``data/raw/``.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache_dir: Path = RAW_DATA_DIR,
    ) -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / _DEFAULT_CACHE_FILE

    # ── public API ──────────────────────────────────────────────────────

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return close prices, downloading only what is missing.

        Parameters
        ----------
        tickers : list[str]
            Requested ticker symbols.
        start : str
            Start date ``YYYY-MM-DD``.
        end : str or None
            End date ``YYYY-MM-DD``. Defaults to today.

        Returns
        -------
        pd.DataFrame
            Close prices with DatetimeIndex rows and ticker columns.
        """
        if end is None:
            end = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        cached = self._read_cache()

        # Determine what needs downloading
        if cached is not None:
            missing_tickers = [t for t in tickers if t not in cached.columns]
            cache_end = cached.index.max()
            # Allow a 3-day gap (weekends/holidays) before triggering a re-download
            needs_date_update = cache_end < (end_dt - pd.Timedelta(days=3))
        else:
            missing_tickers = list(tickers)
            needs_date_update = False

        new_frames: list[pd.DataFrame] = []

        # Download missing tickers (full date range)
        if missing_tickers:
            logger.info("Downloading %d missing tickers: %s", len(missing_tickers), missing_tickers)
            try:
                new_data = self.provider.fetch_prices(missing_tickers, start=start, end=end)
                new_frames.append(new_data)
            except DataError:
                logger.warning("All missing tickers failed to download — skipping.")


        # Download new dates for existing tickers
        if cached is not None and needs_date_update:
            existing_tickers = [t for t in tickers if t in cached.columns]
            if existing_tickers:
                update_start = (cache_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(
                    "Updating %d tickers from %s to %s",
                    len(existing_tickers),
                    update_start,
                    end,
                )
                try:
                    new_dates = self.provider.fetch_prices(existing_tickers, start=update_start, end=end)
                    if not new_dates.empty:
                        new_frames.append(new_dates)
                except DataError:
                    logger.warning("Date update failed for all tickers — skipping.")

        # Merge everything and save
        if new_frames:
            merged = self._merge(cached, new_frames)
            self._write_cache(merged)
            cached = merged

        if cached is None:
            return pd.DataFrame()

        # Slice to requested tickers and date range
        available = [t for t in tickers if t in cached.columns]
        result = cached.loc[start_dt:end_dt, available]
        return result

    def update(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
    ) -> pd.DataFrame:
        """Refresh the cache from the last cached date to today.

        Parameters
        ----------
        tickers : list[str] or None
            Tickers to update. If None, updates all cached tickers.
        start : str or None
            Override start date for new ticker downloads.

        Returns
        -------
        pd.DataFrame
            The full updated cache.
        """
        cached = self._read_cache()
        today = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")

        if tickers is None:
            if cached is None:
                logger.warning("No cache exists and no tickers specified — nothing to update.")
                return pd.DataFrame()
            tickers = list(cached.columns)

        if cached is not None:
            cache_start = cached.index.min().strftime("%Y-%m-%d")
        else:
            cache_start = "2000-01-01"

        effective_start = start if start is not None else cache_start
        return self.get_prices(tickers, start=effective_start, end=today)

    def list_cached(self) -> list[str]:
        """Return list of ticker symbols currently in the cache.

        Returns
        -------
        list[str]
            Sorted list of cached ticker symbols, or empty list if no cache.
        """
        cached = self._read_cache()
        if cached is None:
            return []
        return sorted(cached.columns.tolist())

    # ── private helpers ─────────────────────────────────────────────────

    def _read_cache(self) -> pd.DataFrame | None:
        """Read the Parquet cache file, or return None if it doesn't exist."""
        if not self.cache_path.exists():
            return None
        df = pd.read_parquet(self.cache_path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def _write_cache(self, df: pd.DataFrame) -> None:
        """Write DataFrame to the Parquet cache file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df = df.sort_index()
        df.to_parquet(self.cache_path, engine="pyarrow")
        logger.info(
            "Cache saved: %d rows x %d tickers → %s",
            len(df),
            len(df.columns),
            self.cache_path,
        )

    @staticmethod
    def _merge(
        cached: pd.DataFrame | None,
        new_frames: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge cached data with newly downloaded frames."""
        frames = []
        if cached is not None:
            frames.append(cached)
        frames.extend(new_frames)

        # Concat along columns (new tickers) and rows (new dates), dedup
        combined = pd.concat(frames, axis=1)
        # If a column appears in multiple frames, keep the last (newest download)
        combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
        combined = combined.sort_index()
        # Drop fully-duplicate rows (same date, same values)
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined
