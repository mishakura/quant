"""Tests for quant.data.cleaners."""

import numpy as np
import pandas as pd
import pytest

from quant.data.cleaners import (
    align_to_common_dates,
    find_common_start_date,
    trim_to_common_history,
    validate_price_data,
)
from quant.exceptions import DataError


class TestFindCommonStartDate:
    def test_same_start(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({"A": range(10), "B": range(10)}, index=idx)
        result = find_common_start_date(df)
        assert result == idx[0]

    def test_different_starts(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({"A": range(10), "B": [np.nan, np.nan] + list(range(8))}, index=idx)
        result = find_common_start_date(df)
        assert result == idx[2]

    def test_all_nan_column_raises(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"A": range(5), "B": [np.nan] * 5}, index=idx)
        with pytest.raises(DataError, match="no valid data"):
            find_common_start_date(df)


class TestTrimToCommonHistory:
    def test_trims_nans(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({
            "A": range(10),
            "B": [np.nan, np.nan, np.nan] + list(range(7)),
        }, index=idx)
        result = trim_to_common_history(df)
        assert len(result) == 7
        assert result.index[0] == idx[3]

    def test_no_trim_needed(self, sample_returns_df: pd.DataFrame) -> None:
        result = trim_to_common_history(sample_returns_df)
        assert len(result) == len(sample_returns_df)


class TestAlignToCommonDates:
    def test_inner_join(self) -> None:
        idx1 = pd.bdate_range("2024-01-01", periods=10, freq="B")
        idx2 = pd.bdate_range("2024-01-05", periods=10, freq="B")
        s1 = pd.Series(range(10), index=idx1, name="A")
        s2 = pd.Series(range(10), index=idx2, name="B")
        a1, a2 = align_to_common_dates(s1, s2)
        assert len(a1) == len(a2)
        assert (a1.index == a2.index).all()

    def test_requires_at_least_two(self) -> None:
        s = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="at least 2"):
            align_to_common_dates(s)


class TestValidatePriceData:
    def test_valid_prices(self, sample_prices_df: pd.DataFrame) -> None:
        result = validate_price_data(sample_prices_df)
        assert result is sample_prices_df

    def test_negative_prices_raise(self) -> None:
        idx = pd.bdate_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"A": [100, -5, 102]}, index=idx)
        with pytest.raises(DataError, match="negative"):
            validate_price_data(df)

    def test_duplicate_dates_raise(self) -> None:
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
        df = pd.DataFrame({"A": [100, 101, 102]}, index=idx)
        with pytest.raises(DataError, match="duplicate"):
            validate_price_data(df)
