"""Tests for Universe class and filtering utilities."""

import pytest

from quant.data.universe import US_EXCHANGES, SPDR_SECTOR_ETFS, Universe


class TestUniverse:
    """Tests for the Universe container."""

    def test_constructor(self) -> None:
        u = Universe(name="test", tickers=["A", "B", "C"])
        assert u.name == "test"
        assert len(u) == 3
        assert u.tickers == ["A", "B", "C"]

    def test_empty_universe(self) -> None:
        u = Universe(name="empty", tickers=[])
        assert len(u) == 0

    def test_repr(self) -> None:
        u = Universe(name="demo", tickers=["X", "Y"])
        assert "demo" in repr(u)
        assert "2" in repr(u)

    def test_filters_default_empty(self) -> None:
        u = Universe(name="test", tickers=["A"])
        assert u.filters == {}

    def test_filters_stored(self) -> None:
        u = Universe(
            name="filtered",
            tickers=["A"],
            filters={"min_market_cap": 1e9},
        )
        assert u.filters["min_market_cap"] == 1e9


class TestConstants:
    """Tests for module-level constants."""

    def test_us_exchanges_set(self) -> None:
        assert isinstance(US_EXCHANGES, set)
        assert "NMS" in US_EXCHANGES
        assert "NYQ" in US_EXCHANGES

    def test_spdr_etfs_list(self) -> None:
        assert isinstance(SPDR_SECTOR_ETFS, list)
        assert len(SPDR_SECTOR_ETFS) == 11
        assert "XLK" in SPDR_SECTOR_ETFS
