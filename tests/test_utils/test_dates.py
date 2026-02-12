"""Tests for quant.utils.dates."""

import pandas as pd
import pytest

from quant.utils.dates import (
    get_rebalance_dates,
    infer_periods_per_year,
    is_quarter_end,
    last_quarter_end,
)


class TestIsQuarterEnd:
    def test_march_31(self) -> None:
        assert is_quarter_end(pd.Timestamp("2024-03-31")) is True

    def test_june_30(self) -> None:
        assert is_quarter_end(pd.Timestamp("2024-06-30")) is True

    def test_september_30(self) -> None:
        assert is_quarter_end(pd.Timestamp("2024-09-30")) is True

    def test_december_31(self) -> None:
        assert is_quarter_end(pd.Timestamp("2024-12-31")) is True

    def test_non_quarter_end(self) -> None:
        assert is_quarter_end(pd.Timestamp("2024-01-15")) is False

    def test_string_input(self) -> None:
        assert is_quarter_end("2024-06-30") is True

    def test_mid_quarter_month(self) -> None:
        assert is_quarter_end(pd.Timestamp("2024-03-15")) is False


class TestLastQuarterEnd:
    def test_finds_latest(self) -> None:
        dates = pd.DatetimeIndex([
            "2024-03-31", "2024-06-30", "2024-07-15", "2024-09-30"
        ])
        result = last_quarter_end(dates)
        assert result == pd.Timestamp("2024-09-30")

    def test_raises_when_no_quarter_end(self) -> None:
        dates = pd.DatetimeIndex(["2024-01-15", "2024-02-20", "2024-04-10"])
        with pytest.raises(ValueError, match="No quarter-end dates found"):
            last_quarter_end(dates)


class TestInferPeriodsPerYear:
    def test_daily_frequency(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=100, freq="B")
        assert infer_periods_per_year(dates) == 252

    def test_weekly_frequency(self) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="W")
        assert infer_periods_per_year(dates) == 52

    def test_monthly_frequency(self) -> None:
        dates = pd.date_range("2024-01-31", periods=24, freq="ME")
        assert infer_periods_per_year(dates) == 12

    def test_quarterly_frequency(self) -> None:
        dates = pd.date_range("2024-03-31", periods=8, freq="QE")
        assert infer_periods_per_year(dates) == 4

    def test_single_date_defaults_to_252(self) -> None:
        dates = pd.DatetimeIndex(["2024-01-15"])
        assert infer_periods_per_year(dates) == 252


class TestGetRebalanceDates:
    def test_monthly_rebalance(self) -> None:
        dates = pd.bdate_range("2024-01-01", "2024-06-30", freq="B")
        result = get_rebalance_dates(dates, frequency="monthly")
        # Should have one date per month
        assert len(result) >= 5

    def test_quarterly_rebalance(self) -> None:
        dates = pd.bdate_range("2024-01-01", "2024-12-31", freq="B")
        result = get_rebalance_dates(dates, frequency="quarterly")
        assert len(result) >= 3

    def test_custom_months(self) -> None:
        dates = pd.bdate_range("2024-01-01", "2024-12-31", freq="B")
        result = get_rebalance_dates(dates, months=[2, 5, 8, 11])
        # Should have one date per specified month
        assert len(result) >= 4

    def test_invalid_frequency_raises(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=100, freq="B")
        with pytest.raises(ValueError, match="Unknown frequency"):
            get_rebalance_dates(dates, frequency="biweekly")
