"""Tests for QuantitativeValueFactor."""

import numpy as np
import pandas as pd
import pytest

from quant.exceptions import DataError
from quant.factors.value import (
    QuantitativeValueFactor,
    _compute_ticker_metrics,
    _geometric_mean,
    _percentile_scores,
    _safe_get,
)


# ── Helpers to build synthetic fundamental data ─────────────────────────────


def _make_income(tickers: list[str]) -> pd.DataFrame:
    """Build synthetic income statement with MultiIndex (ticker, field)."""
    dates = pd.to_datetime(["2023-12-31", "2022-12-31", "2021-12-31", "2020-12-31"])
    rows = []
    for ticker in tickers:
        base = hash(ticker) % 100 + 50  # deterministic per-ticker variation
        rows.append({
            "ticker": ticker, "field": "TotalRevenue",
            dates[0]: 1e6 * base, dates[1]: 0.9e6 * base,
            dates[2]: 0.8e6 * base, dates[3]: 0.7e6 * base,
        })
        rows.append({
            "ticker": ticker, "field": "GrossProfit",
            dates[0]: 0.5e6 * base, dates[1]: 0.45e6 * base,
            dates[2]: 0.4e6 * base, dates[3]: 0.35e6 * base,
        })
        rows.append({
            "ticker": ticker, "field": "OperatingIncome",
            dates[0]: 0.3e6 * base, dates[1]: 0.25e6 * base,
            dates[2]: 0.2e6 * base, dates[3]: 0.18e6 * base,
        })
        rows.append({
            "ticker": ticker, "field": "NetIncome",
            dates[0]: 0.2e6 * base, dates[1]: 0.15e6 * base,
            dates[2]: 0.12e6 * base, dates[3]: 0.1e6 * base,
        })
    df = pd.DataFrame(rows).set_index(["ticker", "field"])
    df.columns = pd.to_datetime(df.columns)
    return df


def _make_balance(tickers: list[str]) -> pd.DataFrame:
    """Build synthetic balance sheet with MultiIndex (ticker, field)."""
    dates = pd.to_datetime(["2023-12-31", "2022-12-31", "2021-12-31", "2020-12-31"])
    rows = []
    for ticker in tickers:
        base = hash(ticker) % 100 + 50
        rows.append({
            "ticker": ticker, "field": "TotalAssets",
            dates[0]: 5e6 * base, dates[1]: 4.8e6 * base,
            dates[2]: 4.5e6 * base, dates[3]: 4.2e6 * base,
        })
        rows.append({
            "ticker": ticker, "field": "CurrentAssets",
            dates[0]: 2e6 * base, dates[1]: 1.9e6 * base,
            dates[2]: 1.8e6 * base, dates[3]: 1.7e6 * base,
        })
        rows.append({
            "ticker": ticker, "field": "CurrentLiabilities",
            dates[0]: 1e6 * base, dates[1]: 1.1e6 * base,
            dates[2]: 1e6 * base, dates[3]: 0.9e6 * base,
        })
        rows.append({
            "ticker": ticker, "field": "NetPPE",
            dates[0]: 1.5e6 * base, dates[1]: 1.4e6 * base,
            dates[2]: 1.3e6 * base, dates[3]: 1.2e6 * base,
        })
        rows.append({
            "ticker": ticker, "field": "LongTermDebt",
            dates[0]: 0.5e6 * base, dates[1]: 0.6e6 * base,
            dates[2]: 0.7e6 * base, dates[3]: 0.8e6 * base,
        })
    df = pd.DataFrame(rows).set_index(["ticker", "field"])
    df.columns = pd.to_datetime(df.columns)
    return df


def _make_cashflow(tickers: list[str]) -> pd.DataFrame:
    """Build synthetic cash flow with MultiIndex (ticker, field)."""
    dates = pd.to_datetime(["2023-12-31", "2022-12-31", "2021-12-31", "2020-12-31"])
    rows = []
    for ticker in tickers:
        base = hash(ticker) % 100 + 50
        rows.append({
            "ticker": ticker, "field": "FreeCashFlow",
            dates[0]: 0.15e6 * base, dates[1]: 0.12e6 * base,
            dates[2]: 0.1e6 * base, dates[3]: 0.08e6 * base,
        })
    df = pd.DataFrame(rows).set_index(["ticker", "field"])
    df.columns = pd.to_datetime(df.columns)
    return df


def _make_income_ttm(tickers: list[str]) -> pd.DataFrame:
    """Build synthetic TTM income with MultiIndex (ticker, field)."""
    dates = pd.to_datetime(["2024-06-30"])
    rows = []
    for ticker in tickers:
        base = hash(ticker) % 100 + 50
        rows.append({
            "ticker": ticker, "field": "OperatingIncome",
            dates[0]: 0.35e6 * base,
        })
    df = pd.DataFrame(rows).set_index(["ticker", "field"])
    df.columns = pd.to_datetime(df.columns)
    return df


def _make_ticker_info(tickers: list[str]) -> pd.DataFrame:
    """Build synthetic ticker info DataFrame."""
    records = []
    for i, ticker in enumerate(tickers):
        base = hash(ticker) % 100 + 50
        records.append({
            "ticker": ticker,
            "market_cap": 2e9 + i * 1e9,  # All above 1.4B threshold
            "enterprise_value": 3e9 + i * 1e9,
            "financial_currency": "USD",
            "quote_type": "EQUITY",
        })
    return pd.DataFrame(records).set_index("ticker")


# ── Fixtures ────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM"]


@pytest.fixture
def fundamentals() -> dict[str, pd.DataFrame]:
    return {
        "income": _make_income(TICKERS),
        "balance": _make_balance(TICKERS),
        "cashflow": _make_cashflow(TICKERS),
        "income_ttm": _make_income_ttm(TICKERS),
    }


@pytest.fixture
def ticker_info() -> pd.DataFrame:
    return _make_ticker_info(TICKERS)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Minimal prices DataFrame — only used to define universe."""
    dates = pd.bdate_range("2023-01-02", periods=100)
    rng = np.random.default_rng(42)
    data = {t: 100 + rng.standard_normal(100).cumsum() for t in TICKERS}
    return pd.DataFrame(data, index=dates)


# ── Unit tests for helper functions ─────────────────────────────────────────


class TestSafeGet:

    def test_existing_field(self, fundamentals):
        val = _safe_get(fundamentals["income"], "AAPL", "NetIncome", 0)
        assert val is not None
        assert isinstance(val, float)

    def test_missing_field(self, fundamentals):
        val = _safe_get(fundamentals["income"], "AAPL", "NonExistentField", 0)
        assert val is None

    def test_missing_ticker(self, fundamentals):
        val = _safe_get(fundamentals["income"], "ZZZZ", "NetIncome", 0)
        assert val is None


class TestGeometricMean:

    def test_positive_values(self):
        result = _geometric_mean([0.10, 0.12, 0.08])
        assert result is not None
        assert 0.09 < result < 0.11

    def test_too_few_values(self):
        result = _geometric_mean([0.10, 0.12])
        assert result is None  # needs >= 3

    def test_empty(self):
        assert _geometric_mean([]) is None

    def test_with_nans(self):
        result = _geometric_mean([0.10, np.nan, 0.12, 0.08])
        assert result is not None


class TestPercentileScores:

    def test_basic(self):
        s = pd.Series([10, 20, 30, 40, 50])
        pcts = _percentile_scores(s)
        assert pcts.iloc[0] < pcts.iloc[-1]  # 10 < 50 in percentile
        assert (pcts >= 0).all() and (pcts <= 1).all()

    def test_with_nans(self):
        s = pd.Series([10, np.nan, 30, np.nan, 50])
        pcts = _percentile_scores(s)
        assert pd.isna(pcts.iloc[1])
        assert pd.isna(pcts.iloc[3])
        assert pd.notna(pcts.iloc[0])


# ── Tests for _compute_ticker_metrics ───────────────────────────────────────


class TestComputeTickerMetrics:

    def test_returns_dict_with_ticker(self, fundamentals, ticker_info):
        metrics = _compute_ticker_metrics(
            "AAPL",
            fundamentals["income"],
            fundamentals["balance"],
            fundamentals["cashflow"],
            fundamentals["income_ttm"],
            ticker_info.loc["AAPL"],
        )
        assert metrics["ticker"] == "AAPL"

    def test_roa_computed(self, fundamentals, ticker_info):
        metrics = _compute_ticker_metrics(
            "AAPL",
            fundamentals["income"],
            fundamentals["balance"],
            fundamentals["cashflow"],
            fundamentals["income_ttm"],
            ticker_info.loc["AAPL"],
        )
        assert metrics.get("eight_year_roa") is not None

    def test_fscore_flags_are_binary(self, fundamentals, ticker_info):
        metrics = _compute_ticker_metrics(
            "AAPL",
            fundamentals["income"],
            fundamentals["balance"],
            fundamentals["cashflow"],
            fundamentals["income_ttm"],
            ticker_info.loc["AAPL"],
        )
        fs_keys = [k for k in metrics if k.startswith("fs_")]
        for key in fs_keys:
            assert metrics[key] in (0, 1), f"{key} = {metrics[key]} is not binary"

    def test_missing_ticker_returns_minimal(self, fundamentals, ticker_info):
        metrics = _compute_ticker_metrics(
            "NONEXISTENT",
            fundamentals["income"],
            fundamentals["balance"],
            fundamentals["cashflow"],
            fundamentals["income_ttm"],
            None,
        )
        assert metrics["ticker"] == "NONEXISTENT"
        # Should not have computed ROA etc.
        assert "eight_year_roa" not in metrics

    def test_ebit_ev_computed(self, fundamentals, ticker_info):
        metrics = _compute_ticker_metrics(
            "AAPL",
            fundamentals["income"],
            fundamentals["balance"],
            fundamentals["cashflow"],
            fundamentals["income_ttm"],
            ticker_info.loc["AAPL"],
        )
        assert metrics.get("ebit_ev") is not None
        assert metrics["ebit_ev"] > 0


# ── Tests for QuantitativeValueFactor ───────────────────────────────────────


class TestQuantitativeValueFactor:

    def test_compute_basic(self, prices, fundamentals, ticker_info):
        factor = QuantitativeValueFactor(min_market_cap=0, usd_only=False, ebit_ev_top_pct=1.0)
        result = factor.compute(
            prices,
            fundamentals=fundamentals,
            ticker_info=ticker_info,
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"ticker", "qv_score", "quality", "franchise_power",
                         "financial_strength", "ebit_ev", "score", "rank"}
        assert expected_cols <= set(result.columns)
        assert len(result) > 0

    def test_score_equals_qv_score(self, prices, fundamentals, ticker_info):
        factor = QuantitativeValueFactor(min_market_cap=0, usd_only=False, ebit_ev_top_pct=1.0)
        result = factor.compute(
            prices,
            fundamentals=fundamentals,
            ticker_info=ticker_info,
        )
        np.testing.assert_allclose(result["score"].values, result["qv_score"].values)

    def test_rank_range(self, prices, fundamentals, ticker_info):
        factor = QuantitativeValueFactor(min_market_cap=0, usd_only=False, ebit_ev_top_pct=1.0)
        result = factor.compute(
            prices,
            fundamentals=fundamentals,
            ticker_info=ticker_info,
        )
        assert (result["rank"] >= 0).all()
        assert (result["rank"] <= 100).all()

    def test_sorted_by_rank_descending(self, prices, fundamentals, ticker_info):
        factor = QuantitativeValueFactor(min_market_cap=0, usd_only=False, ebit_ev_top_pct=1.0)
        result = factor.compute(
            prices,
            fundamentals=fundamentals,
            ticker_info=ticker_info,
        )
        assert result["rank"].is_monotonic_decreasing or len(result) <= 1

    def test_quality_composition(self, prices, fundamentals, ticker_info):
        """QUALITY = 0.5 * franchise_power + 0.5 * financial_strength."""
        factor = QuantitativeValueFactor(min_market_cap=0, usd_only=False, ebit_ev_top_pct=1.0)
        result = factor.compute(
            prices,
            fundamentals=fundamentals,
            ticker_info=ticker_info,
        )
        expected = 0.5 * result["franchise_power"] + 0.5 * result["financial_strength"]
        np.testing.assert_allclose(result["quality"].values, expected.values, atol=1e-10)

    def test_ebit_ev_filter(self, prices, fundamentals, ticker_info):
        """With ebit_ev_top_pct=0.5, fewer tickers should pass."""
        factor_all = QuantitativeValueFactor(min_market_cap=0, usd_only=False, ebit_ev_top_pct=1.0)
        result_all = factor_all.compute(prices, fundamentals=fundamentals, ticker_info=ticker_info)

        factor_half = QuantitativeValueFactor(min_market_cap=0, usd_only=False, ebit_ev_top_pct=0.5)
        result_half = factor_half.compute(prices, fundamentals=fundamentals, ticker_info=ticker_info)

        assert len(result_half) <= len(result_all)

    def test_market_cap_filter(self, prices, fundamentals, ticker_info):
        """High min_market_cap should filter out tickers."""
        factor = QuantitativeValueFactor(min_market_cap=100e9, usd_only=False, ebit_ev_top_pct=1.0)
        with pytest.raises(DataError, match="No tickers passed pre-filters"):
            factor.compute(prices, fundamentals=fundamentals, ticker_info=ticker_info)

    def test_usd_filter(self, prices, fundamentals):
        """Non-USD tickers should be filtered out when usd_only=True."""
        info = _make_ticker_info(TICKERS)
        # Set half to non-USD
        for i, ticker in enumerate(TICKERS):
            if i % 2 == 0:
                info.loc[ticker, "financial_currency"] = "EUR"

        factor = QuantitativeValueFactor(min_market_cap=0, usd_only=True, ebit_ev_top_pct=1.0)
        result = factor.compute(prices, fundamentals=fundamentals, ticker_info=info)

        # All remaining tickers should be USD ones
        usd_tickers = set(info[info["financial_currency"] == "USD"].index)
        assert set(result["ticker"]).issubset(usd_tickers)

    def test_missing_fundamentals_raises(self, prices, ticker_info):
        factor = QuantitativeValueFactor()
        with pytest.raises(DataError, match="requires 'fundamentals'"):
            factor.compute(prices, ticker_info=ticker_info)

    def test_missing_ticker_info_raises(self, prices, fundamentals):
        factor = QuantitativeValueFactor()
        with pytest.raises(DataError, match="requires 'ticker_info'"):
            factor.compute(prices, fundamentals=fundamentals)

    def test_requires_fundamentals_flag(self):
        factor = QuantitativeValueFactor()
        assert factor.requires_fundamentals is True

    def test_financial_strength_bounded(self, prices, fundamentals, ticker_info):
        """Financial strength should be between 0 and 1."""
        factor = QuantitativeValueFactor(min_market_cap=0, usd_only=False, ebit_ev_top_pct=1.0)
        result = factor.compute(prices, fundamentals=fundamentals, ticker_info=ticker_info)
        assert (result["financial_strength"] >= 0).all()
        assert (result["financial_strength"] <= 1).all()

    def test_registered_name(self):
        """Factor is registered under 'quantitative_value'."""
        from quant.factors.registry import get_factor
        cls = get_factor("quantitative_value")
        assert cls is QuantitativeValueFactor
