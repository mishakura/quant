"""Tests for quant.data.fundamentals — FundamentalStore Parquet cache."""

import numpy as np
import pandas as pd
import pytest

from quant.data.fundamentals import FundamentalStore, _STATEMENT_FILES


# ── Fake yfinance Ticker for testing ────────────────────────────────────────


class FakeYFTicker:
    """Minimal yfinance.Ticker stand-in returning deterministic statements."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self._dates = pd.to_datetime(["2023-12-31", "2022-12-31", "2021-12-31"])
        self._ttm_dates = pd.to_datetime(["2024-06-30"])

    def get_income_stmt(self, freq: str = "yearly") -> pd.DataFrame:
        if freq == "trailing":
            return pd.DataFrame(
                {"OperatingIncome": [500_000.0], "NetIncome": [400_000.0]},
                index=self._ttm_dates,
            ).T
        return pd.DataFrame(
            {
                self._dates[0]: [1_000_000, 800_000, 600_000, 200_000],
                self._dates[1]: [900_000, 720_000, 540_000, 180_000],
                self._dates[2]: [800_000, 640_000, 480_000, 160_000],
            },
            index=["TotalRevenue", "GrossProfit", "OperatingIncome", "NetIncome"],
        )

    def get_balance_sheet(self, freq: str = "yearly") -> pd.DataFrame:
        return pd.DataFrame(
            {
                self._dates[0]: [5_000_000, 2_000_000, 1_000_000, 1_500_000, 500_000],
                self._dates[1]: [4_800_000, 1_900_000, 950_000, 1_400_000, 600_000],
                self._dates[2]: [4_600_000, 1_800_000, 900_000, 1_300_000, 700_000],
            },
            index=["TotalAssets", "CurrentAssets", "CurrentLiabilities", "NetPPE", "LongTermDebt"],
        )

    def get_cashflow(self, freq: str = "yearly") -> pd.DataFrame:
        return pd.DataFrame(
            {
                self._dates[0]: [300_000.0],
                self._dates[1]: [250_000.0],
                self._dates[2]: [200_000.0],
            },
            index=["FreeCashFlow"],
        )


# ── Monkeypatch helper ──────────────────────────────────────────────────────


def _fake_ticker_factory(ticker: str) -> FakeYFTicker:
    return FakeYFTicker(ticker)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def fund_store(tmp_path, monkeypatch):
    """FundamentalStore that uses fake yfinance and writes to tmp dir."""
    monkeypatch.setattr("quant.data.fundamentals.yf.Ticker", _fake_ticker_factory)
    return FundamentalStore(cache_dir=tmp_path, sleep=0.0)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestFundamentalStore:
    """FundamentalStore core functionality."""

    def test_update_downloads_and_caches(self, fund_store):
        """update() downloads data and writes Parquet files."""
        result = fund_store.update(["AAPL", "MSFT"])

        assert "income" in result
        assert "balance" in result
        assert "cashflow" in result
        assert "income_ttm" in result

        # Check income statement has data for both tickers
        income = result["income"]
        assert not income.empty
        tickers_in_cache = income.index.get_level_values(0).unique().tolist()
        assert "AAPL" in tickers_in_cache
        assert "MSFT" in tickers_in_cache

        # Verify Parquet files exist
        for filename in _STATEMENT_FILES.values():
            assert (fund_store.cache_dir / filename).exists()

    def test_update_merges_new_tickers(self, fund_store):
        """Second update() adds new tickers without losing old ones."""
        fund_store.update(["AAPL"])
        fund_store.update(["MSFT"])

        cached = fund_store.list_cached()
        assert "AAPL" in cached
        assert "MSFT" in cached

    def test_load_all(self, fund_store):
        """load() returns all cached data when no filters specified."""
        fund_store.update(["AAPL"])
        result = fund_store.load()

        assert "income" in result
        assert not result["income"].empty

    def test_load_specific_tickers(self, fund_store):
        """load(tickers=...) filters to requested tickers only."""
        fund_store.update(["AAPL", "MSFT"])
        result = fund_store.load(tickers=["AAPL"])

        income = result["income"]
        tickers_loaded = income.index.get_level_values(0).unique().tolist()
        assert tickers_loaded == ["AAPL"]

    def test_load_specific_statements(self, fund_store):
        """load(statements=...) returns only requested statement types."""
        fund_store.update(["AAPL"])
        result = fund_store.load(statements=["income", "balance"])

        assert "income" in result
        assert "balance" in result
        assert "cashflow" not in result
        assert "income_ttm" not in result

    def test_list_cached_empty(self, fund_store):
        """list_cached() returns empty list when no cache exists."""
        assert fund_store.list_cached() == []

    def test_list_cached_returns_tickers(self, fund_store):
        """list_cached() returns sorted list of cached tickers."""
        fund_store.update(["MSFT", "AAPL"])
        assert fund_store.list_cached() == ["AAPL", "MSFT"]

    def test_update_empty_tickers(self, fund_store):
        """update([]) returns empty DataFrames."""
        result = fund_store.update([])
        for df in result.values():
            assert df.empty

    def test_multiindex_structure(self, fund_store):
        """Cached data has MultiIndex (ticker, field)."""
        fund_store.update(["AAPL"])
        result = fund_store.load()

        income = result["income"]
        assert income.index.names == ["ticker", "field"]
        assert "NetIncome" in income.index.get_level_values("field")

    def test_load_missing_tickers_returns_empty(self, fund_store):
        """load() with tickers not in cache returns empty DataFrames."""
        fund_store.update(["AAPL"])
        result = fund_store.load(tickers=["NONEXISTENT"])

        for df in result.values():
            assert df.empty


class TestLoadFundamentals:
    """Tests for the load_fundamentals() convenience function."""

    def test_load_from_cache(self, fund_store):
        """load_fundamentals() reads from existing cache."""
        from quant.data.loaders import load_fundamentals

        fund_store.update(["AAPL"])
        result = load_fundamentals(cache_dir=fund_store.cache_dir)

        assert "income" in result
        assert not result["income"].empty

    def test_load_missing_cache(self, tmp_path):
        """load_fundamentals() raises DataError when no cache exists."""
        from quant.data.loaders import load_fundamentals
        from quant.exceptions import DataError

        with pytest.raises(DataError, match="No cached fundamental data"):
            load_fundamentals(cache_dir=tmp_path / "nonexistent")
