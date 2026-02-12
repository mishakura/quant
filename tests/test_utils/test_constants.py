"""Tests for quant.utils.constants."""

from quant.utils.constants import (
    DEFAULT_RISK_FREE_RATE,
    QUARTER_END_DATES,
    RISK_MEASURES,
    TRADING_DAYS_PER_YEAR,
)


def test_trading_days_is_252() -> None:
    assert TRADING_DAYS_PER_YEAR == 252


def test_risk_free_rate_is_positive() -> None:
    assert DEFAULT_RISK_FREE_RATE > 0


def test_risk_measures_has_expected_count() -> None:
    assert len(RISK_MEASURES) == 13


def test_risk_measures_contains_core_measures() -> None:
    core = {"MV", "CVaR", "VaR", "MDD", "CDaR", "UCI"}
    assert core.issubset(set(RISK_MEASURES))


def test_quarter_end_dates_has_four_entries() -> None:
    assert len(QUARTER_END_DATES) == 4


def test_quarter_end_dates_are_valid() -> None:
    for month, day in QUARTER_END_DATES:
        assert 1 <= month <= 12
        assert 1 <= day <= 31
