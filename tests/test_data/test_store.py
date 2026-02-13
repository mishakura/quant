"""Tests for quant.data.store — PriceStore local Parquet cache."""

import pandas as pd
import pytest

from quant.data.store import PriceStore
from quant.exceptions import DataError


# ── Fake provider for testing ───────────────────────────────────────────────


class FakeProvider:
    """Minimal DataProvider stand-in that returns deterministic prices."""

    def __init__(self) -> None:
        self.call_log: list[dict] = []

    def fetch_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        self.call_log.append({"tickers": tickers, "start": start, "end": end})
        dates = pd.bdate_range(start=start, end=end or "2024-06-30")
        data = {t: range(100, 100 + len(dates)) for t in tickers}
        return pd.DataFrame(data, index=dates)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path):
    """PriceStore backed by a FakeProvider writing to a temp directory."""
    provider = FakeProvider()
    return PriceStore(provider, cache_dir=tmp_path)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestPriceStore:
    """PriceStore core functionality."""

    def test_get_prices_downloads_and_caches(self, store):
        """First call downloads, second call reads from cache."""
        prices = store.get_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-06-30")

        assert isinstance(prices, pd.DataFrame)
        assert "AAPL" in prices.columns
        assert "MSFT" in prices.columns
        assert store.cache_path.exists()

        # Second call should NOT trigger a new download
        provider = store.provider
        call_count_before = len(provider.call_log)
        prices2 = store.get_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-06-30")
        assert len(provider.call_log) == call_count_before
        pd.testing.assert_frame_equal(prices, prices2, check_freq=False)

    def test_get_prices_downloads_only_missing_tickers(self, store):
        """When some tickers are cached, only missing ones are downloaded."""
        store.get_prices(["AAPL"], start="2024-01-01", end="2024-06-30")
        call_count_after_first = len(store.provider.call_log)

        store.get_prices(["AAPL", "SPY"], start="2024-01-01", end="2024-06-30")
        # Should have made one more call, only for SPY
        assert len(store.provider.call_log) == call_count_after_first + 1
        last_call = store.provider.call_log[-1]
        assert last_call["tickers"] == ["SPY"]

    def test_get_prices_updates_dates(self, store):
        """Cache with old end date triggers a date update download."""
        store.get_prices(["AAPL"], start="2024-01-01", end="2024-03-31")

        # Now request up to June — should trigger an update
        call_count_before = len(store.provider.call_log)
        store.get_prices(["AAPL"], start="2024-01-01", end="2024-06-30")
        assert len(store.provider.call_log) > call_count_before

    def test_list_cached_empty(self, store):
        """list_cached returns empty list when no cache exists."""
        assert store.list_cached() == []

    def test_list_cached_returns_tickers(self, store):
        """list_cached returns sorted ticker list after caching."""
        store.get_prices(["MSFT", "AAPL"], start="2024-01-01", end="2024-06-30")
        assert store.list_cached() == ["AAPL", "MSFT"]

    def test_update_refreshes_all_cached(self, store):
        """update() with no args refreshes all cached tickers."""
        store.get_prices(["AAPL", "SPY"], start="2024-01-01", end="2024-03-31")
        call_count_before = len(store.provider.call_log)

        store.update()
        # Should have made at least one new call
        assert len(store.provider.call_log) > call_count_before

    def test_update_with_specific_tickers(self, store):
        """update() with tickers downloads those tickers."""
        store.update(tickers=["GOOG"], start="2024-01-01")
        assert "GOOG" in store.list_cached()

    def test_update_empty_cache_no_tickers(self, store):
        """update() with no cache and no tickers returns empty DataFrame."""
        result = store.update()
        assert result.empty


class TestLoadPrices:
    """Tests for the load_prices() convenience function."""

    def test_load_prices_from_cache(self, tmp_path):
        """load_prices reads from an existing Parquet cache."""
        from quant.data.loaders import load_prices

        # Create a fake cache file
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        df = pd.DataFrame({"AAPL": range(len(dates)), "MSFT": range(len(dates))}, index=dates)
        cache_path = tmp_path / "prices.parquet"
        df.to_parquet(cache_path)

        prices = load_prices(["AAPL"], start="2024-03-01", end="2024-04-30", cache_path=cache_path)
        assert "AAPL" in prices.columns
        assert prices.index.min() >= pd.Timestamp("2024-03-01")
        assert prices.index.max() <= pd.Timestamp("2024-04-30")

    def test_load_prices_missing_cache(self, tmp_path):
        """load_prices raises DataError when cache file doesn't exist."""
        from quant.data.loaders import load_prices

        with pytest.raises(DataError, match="No cached price data"):
            load_prices(["AAPL"], cache_path=tmp_path / "nonexistent.parquet")

    def test_load_prices_warns_missing_tickers(self, tmp_path):
        """load_prices warns when requested tickers aren't in the cache."""
        from quant.data.loaders import load_prices

        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        df = pd.DataFrame({"AAPL": range(len(dates))}, index=dates)
        cache_path = tmp_path / "prices.parquet"
        df.to_parquet(cache_path)

        prices = load_prices(["AAPL", "GOOG"], cache_path=cache_path)
        assert "AAPL" in prices.columns
        assert "GOOG" not in prices.columns
