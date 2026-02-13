"""Tests for SectorMomentumFactor."""

import numpy as np
import pandas as pd
import pytest

from quant.exceptions import DataError, InsufficientDataError
from quant.factors.sector_rotation import SPDR_SECTOR_ETFS, SectorMomentumFactor


class TestSectorMomentumFactor:
    """Tests for the SPDR sector ETF rotation factor."""

    def test_compute_basic(self, sample_sector_etf_prices: pd.DataFrame) -> None:
        """Factor produces valid output with 11 ETFs."""
        factor = SectorMomentumFactor()
        result = factor.compute(sample_sector_etf_prices)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"ticker", "return", "score", "rank"}
        assert len(result) == 3  # default top_n

    def test_top_n_selection(self, sample_sector_etf_prices: pd.DataFrame) -> None:
        """Output contains exactly top_n rows."""
        factor = SectorMomentumFactor(top_n=5)
        result = factor.compute(sample_sector_etf_prices)
        assert len(result) == 5

    def test_top_1(self, sample_sector_etf_prices: pd.DataFrame) -> None:
        """top_n=1 returns single best sector."""
        factor = SectorMomentumFactor(top_n=1)
        result = factor.compute(sample_sector_etf_prices)
        assert len(result) == 1

    def test_insufficient_data(self, short_prices: pd.DataFrame) -> None:
        """Short data raises InsufficientDataError."""
        factor = SectorMomentumFactor(lookback=231)
        with pytest.raises(InsufficientDataError):
            factor.compute(short_prices)

    def test_missing_etfs_still_works(self) -> None:
        """Factor works with subset of ETFs."""
        rng = np.random.default_rng(55)
        dates = pd.bdate_range("2023-01-02", periods=250, freq="B")
        # Only 5 of 11 ETFs
        etfs = ["XLK", "XLF", "XLE", "XLV", "XLU"]
        returns = np.column_stack([
            rng.normal(0.0003 * (i + 1), 0.01, 250)
            for i in range(len(etfs))
        ])
        prices = pd.DataFrame(
            100.0 * np.cumprod(1 + returns, axis=0),
            index=dates, columns=etfs,
        )
        factor = SectorMomentumFactor(top_n=3)
        result = factor.compute(prices)
        assert len(result) == 3
        # All tickers in result should be from available ETFs
        assert set(result["ticker"]).issubset(set(etfs))

    def test_no_etf_columns_raises(self) -> None:
        """No SPDR ETFs in data raises DataError."""
        dates = pd.bdate_range("2023-01-02", periods=250, freq="B")
        prices = pd.DataFrame(
            np.random.default_rng(1).normal(100, 5, (250, 2)),
            index=dates, columns=["AAPL", "MSFT"],
        )
        factor = SectorMomentumFactor()
        with pytest.raises(DataError):
            factor.compute(prices)

    def test_spdr_list_has_eleven(self) -> None:
        """SPDR_SECTOR_ETFS constant has 11 entries."""
        assert len(SPDR_SECTOR_ETFS) == 11
