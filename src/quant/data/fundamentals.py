"""Local Parquet cache for fundamental financial statements.

Wraps yfinance to download annual income statements, balance sheets,
cash-flow statements, and trailing-twelve-month income data.  Subsequent
calls read from four Parquet files in ``data/raw/``.

The storage layout uses a MultiIndex ``(ticker, field)`` as rows and
period-end dates as columns, mirroring the format returned by yfinance.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from quant.config import RAW_DATA_DIR
from quant.utils.logging import get_logger

logger = get_logger(__name__)

_STATEMENT_FILES: dict[str, str] = {
    "income": "fundamentals_income.parquet",
    "balance": "fundamentals_balance.parquet",
    "cashflow": "fundamentals_cashflow.parquet",
    "income_ttm": "fundamentals_income_ttm.parquet",
}

_SLEEP_SECONDS: float = 0.5


class FundamentalStore:
    """Persistent Parquet cache for financial statements.

    Parameters
    ----------
    cache_dir : Path
        Directory where the Parquet cache files are stored.
        Defaults to ``data/raw/``.
    sleep : float
        Seconds to pause between ticker downloads to avoid throttling.
    """

    def __init__(
        self,
        cache_dir: Path = RAW_DATA_DIR,
        sleep: float = _SLEEP_SECONDS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.sleep = sleep

    # ── public API ──────────────────────────────────────────────────────

    def update(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Download fundamental data for *tickers* and merge into cache.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to download.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of statement name to the full updated cache DataFrame.
        """
        if not tickers:
            logger.warning("No tickers provided — nothing to update.")
            return {name: pd.DataFrame() for name in _STATEMENT_FILES}

        cached = self._read_all()

        new_data: dict[str, list[pd.DataFrame]] = {name: [] for name in _STATEMENT_FILES}
        failed: list[str] = []

        for i, ticker in enumerate(tickers):
            logger.info("Downloading fundamentals %d/%d: %s", i + 1, len(tickers), ticker)
            try:
                data = self._download_ticker(ticker)
                for name, df in data.items():
                    if df is not None and not df.empty:
                        new_data[name].append(df)
            except Exception:
                logger.warning("Failed to download fundamentals for %s", ticker, exc_info=True)
                failed.append(ticker)

            if i < len(tickers) - 1:
                time.sleep(self.sleep)

        if failed:
            logger.warning(
                "Failed to download %d/%d tickers: %s",
                len(failed), len(tickers), failed,
            )

        # Merge new data into cache and save
        result: dict[str, pd.DataFrame] = {}
        for name in _STATEMENT_FILES:
            frames = []
            if cached[name] is not None:
                frames.append(cached[name])
            frames.extend(new_data[name])

            if frames:
                merged = self._merge_frames(frames)
                self._write_cache(name, merged)
                result[name] = merged
            else:
                result[name] = pd.DataFrame()

        total = sum(1 for n in new_data for _ in new_data[n])
        logger.info(
            "Fundamental update complete: %d tickers requested, %d failed",
            len(tickers), len(failed),
        )
        return result

    def load(
        self,
        tickers: list[str] | None = None,
        statements: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load cached fundamental data from disk.

        Parameters
        ----------
        tickers : list[str] or None
            Filter to these tickers. If None, loads all.
        statements : list[str] or None
            Which statements to load (e.g. ``['income', 'balance']``).
            If None, loads all four.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of statement name to DataFrame.
        """
        names = statements if statements is not None else list(_STATEMENT_FILES)
        result: dict[str, pd.DataFrame] = {}

        for name in names:
            if name not in _STATEMENT_FILES:
                logger.warning("Unknown statement type: %s — skipping", name)
                continue
            df = self._read_cache(name)
            if df is None:
                result[name] = pd.DataFrame()
                continue
            if tickers is not None:
                # Filter to requested tickers in the MultiIndex
                available = [t for t in tickers if t in df.index.get_level_values(0)]
                if available:
                    df = df.loc[available]
                else:
                    df = pd.DataFrame()
            result[name] = df

        return result

    def list_cached(self) -> list[str]:
        """Return sorted list of tickers present in the income statement cache.

        Returns
        -------
        list[str]
            Sorted ticker symbols, or empty list if no cache.
        """
        df = self._read_cache("income")
        if df is None or df.empty:
            return []
        return sorted(df.index.get_level_values(0).unique().tolist())

    # ── private helpers ─────────────────────────────────────────────────

    def _download_ticker(self, ticker: str) -> dict[str, pd.DataFrame | None]:
        """Download all four statement types for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        dict[str, pd.DataFrame | None]
            Statement DataFrames with MultiIndex ``(ticker, field)``.
        """
        t = yf.Ticker(ticker)

        result: dict[str, pd.DataFrame | None] = {}
        fetchers = {
            "income": t.get_income_stmt,
            "balance": t.get_balance_sheet,
            "cashflow": t.get_cashflow,
        }

        for name, fetcher in fetchers.items():
            try:
                df = fetcher(freq="yearly")
                if df is not None and not df.empty:
                    result[name] = self._add_ticker_index(df, ticker)
                else:
                    result[name] = None
            except Exception:
                logger.debug("Failed to fetch %s for %s", name, ticker)
                result[name] = None

        # TTM income
        try:
            df = t.get_income_stmt(freq="trailing")
            if df is not None and not df.empty:
                result["income_ttm"] = self._add_ticker_index(df, ticker)
            else:
                result["income_ttm"] = None
        except Exception:
            logger.debug("Failed to fetch income_ttm for %s", ticker)
            result["income_ttm"] = None

        return result

    @staticmethod
    def _add_ticker_index(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Convert yfinance statement DataFrame to MultiIndex (ticker, field).

        yfinance returns statements with fields as rows and dates as columns.
        We prepend a ticker level to the index.
        """
        df = df.copy()
        df.index = pd.MultiIndex.from_product(
            [[ticker], df.index], names=["ticker", "field"],
        )
        # Ensure columns are Timestamps
        df.columns = pd.to_datetime(df.columns)
        return df

    @staticmethod
    def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple statement DataFrames, keeping the latest data per ticker."""
        combined = pd.concat(frames)
        # Remove duplicate (ticker, field) rows, keeping the last occurrence
        combined = combined[~combined.index.duplicated(keep="last")]
        # Union of all date columns, sort chronologically
        combined = combined.reindex(sorted(combined.columns, reverse=True), axis=1)
        return combined

    def _read_cache(self, statement: str) -> pd.DataFrame | None:
        """Read a single statement cache file."""
        path = self.cache_dir / _STATEMENT_FILES[statement]
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        # Restore MultiIndex if stored flat
        if df.index.names == [None] or df.index.names != ["ticker", "field"]:
            if "ticker" in df.columns and "field" in df.columns:
                df = df.set_index(["ticker", "field"])
        return df

    def _read_all(self) -> dict[str, pd.DataFrame | None]:
        """Read all four cache files."""
        return {name: self._read_cache(name) for name in _STATEMENT_FILES}

    def _write_cache(self, statement: str, df: pd.DataFrame) -> None:
        """Write a statement DataFrame to its Parquet cache file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / _STATEMENT_FILES[statement]
        df.to_parquet(path, engine="pyarrow")
        n_tickers = df.index.get_level_values(0).nunique() if not df.empty else 0
        logger.info(
            "Fundamentals cache saved: %s — %d tickers → %s",
            statement, n_tickers, path,
        )
