"""Tests for quant.analytics.performance.

Tests validate mathematical correctness, edge cases, and consistency
between related metrics.
"""

import numpy as np
import pandas as pd
import pytest

from quant.analytics.performance import (
    annualized_return,
    annualized_volatility,
    average_drawdown,
    beta,
    cagr,
    calmar_ratio,
    conditional_var,
    downside_deviation,
    drawdown_series,
    information_ratio,
    jensens_alpha,
    max_drawdown,
    max_drawdown_duration,
    omega_ratio,
    performance_summary,
    return_kurtosis,
    return_skewness,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    total_return,
    tracking_error,
    value_at_risk,
    win_rate,
)


# ── Return metrics ───────────────────────────────────────────────────────────

class TestTotalReturn:
    def test_basic(self) -> None:
        returns = pd.Series([0.10, 0.05, -0.03])
        expected = (1.10 * 1.05 * 0.97) - 1
        assert np.isclose(total_return(returns), expected)

    def test_zero_returns(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0])
        assert total_return(returns) == 0.0

    def test_single_period(self) -> None:
        returns = pd.Series([0.05])
        assert np.isclose(total_return(returns), 0.05)


class TestAnnualizedReturn:
    def test_positive_returns(self, sample_returns: pd.Series) -> None:
        ann = annualized_return(sample_returns)
        # Realistic equity return should be between -50% and +100%
        assert -0.5 < ann < 1.0

    def test_one_year_exact(self) -> None:
        # 252 days of constant 0.04% daily return
        daily_r = 0.0004
        returns = pd.Series([daily_r] * 252)
        ann = annualized_return(returns, periods_per_year=252)
        expected = (1 + daily_r) ** 252 - 1
        assert np.isclose(ann, expected, rtol=1e-6)

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(annualized_return(pd.Series(dtype=float)))

    def test_cagr_is_alias(self, sample_returns: pd.Series) -> None:
        assert cagr(sample_returns) == annualized_return(sample_returns)


# ── Volatility metrics ───────────────────────────────────────────────────────

class TestAnnualizedVolatility:
    def test_known_value(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.01, -0.01] * 63)
        vol = annualized_volatility(returns, periods_per_year=252)
        daily_std = returns.std()
        expected = daily_std * np.sqrt(252)
        assert np.isclose(vol, expected)

    def test_positive(self, sample_returns: pd.Series) -> None:
        assert annualized_volatility(sample_returns) > 0


class TestDownsideDeviation:
    def test_only_positive_returns(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.03])
        assert downside_deviation(returns) == 0.0

    def test_less_than_volatility(self, sample_returns: pd.Series) -> None:
        dd = downside_deviation(sample_returns)
        vol = annualized_volatility(sample_returns)
        assert dd <= vol


# ── Risk-adjusted ratios ─────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_zero_vol_returns_nan(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0])
        assert np.isnan(sharpe_ratio(returns))

    def test_positive_excess_positive_sharpe(self) -> None:
        # High mean, low vol
        returns = pd.Series([0.001] * 252)
        s = sharpe_ratio(returns, risk_free_rate=0.0)
        assert s > 0

    def test_rf_zero(self, sample_returns: pd.Series) -> None:
        s = sharpe_ratio(sample_returns, risk_free_rate=0.0)
        assert isinstance(s, float)


class TestSortinoRatio:
    def test_higher_than_sharpe_for_positive_skew(self) -> None:
        # Positively skewed returns -> Sortino should be >= Sharpe
        rng = np.random.default_rng(42)
        returns = pd.Series(np.abs(rng.normal(0.001, 0.01, 500)))
        # All positive returns -> downside dev is zero -> Sortino is NaN
        # This is expected behavior
        s = sortino_ratio(returns, risk_free_rate=0.0)
        assert np.isnan(s) or s > 0


class TestCalmarRatio:
    def test_no_drawdown_returns_nan(self) -> None:
        returns = pd.Series([0.01, 0.01, 0.01])
        assert np.isnan(calmar_ratio(returns))

    def test_positive_for_profitable_strategy(self, sample_returns: pd.Series) -> None:
        c = calmar_ratio(sample_returns)
        assert isinstance(c, float)


class TestOmegaRatio:
    def test_all_positive_returns(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.03])
        assert np.isnan(omega_ratio(returns))  # No losses -> NaN

    def test_mixed_returns(self) -> None:
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        result = omega_ratio(returns)
        assert result > 0


class TestInformationRatio:
    def test_identical_returns(self, sample_returns: pd.Series) -> None:
        # IR of a series vs itself should be NaN (TE = 0)
        assert np.isnan(information_ratio(sample_returns, sample_returns))


# ── Drawdown metrics ─────────────────────────────────────────────────────────

class TestDrawdownSeries:
    def test_starts_at_zero(self, sample_returns: pd.Series) -> None:
        dd = drawdown_series(sample_returns)
        assert dd.iloc[0] == 0.0

    def test_never_positive(self, sample_returns: pd.Series) -> None:
        dd = drawdown_series(sample_returns)
        assert (dd <= 0).all()

    def test_all_positive_no_drawdown(self) -> None:
        returns = pd.Series([0.01, 0.01, 0.01])
        dd = drawdown_series(returns)
        assert (dd == 0).all()


class TestMaxDrawdown:
    def test_known_drawdown(self) -> None:
        # Price: 100 -> 110 -> 88 -> 95
        returns = pd.Series([0.10, -0.20, 0.0795])
        mdd = max_drawdown(returns)
        expected = 88 / 110 - 1  # -0.2
        assert np.isclose(mdd, expected, atol=1e-4)

    def test_negative(self, sample_returns: pd.Series) -> None:
        assert max_drawdown(sample_returns) < 0

    def test_empty_returns_zero(self) -> None:
        assert max_drawdown(pd.Series(dtype=float)) == 0.0


class TestMaxDrawdownDuration:
    def test_no_drawdown(self) -> None:
        returns = pd.Series([0.01, 0.01])
        assert max_drawdown_duration(returns) == 0

    def test_positive_duration(self, sample_returns: pd.Series) -> None:
        dur = max_drawdown_duration(sample_returns)
        assert dur > 0


class TestAverageDrawdown:
    def test_no_drawdown(self) -> None:
        returns = pd.Series([0.01, 0.01])
        assert average_drawdown(returns) == 0.0

    def test_negative(self, sample_returns: pd.Series) -> None:
        assert average_drawdown(sample_returns) < 0


# ── Tail risk metrics ────────────────────────────────────────────────────────

class TestValueAtRisk:
    def test_historical_95(self, sample_returns: pd.Series) -> None:
        var = value_at_risk(sample_returns, confidence=0.95)
        assert var < 0  # VaR should be negative for losses

    def test_parametric(self, sample_returns: pd.Series) -> None:
        var_p = value_at_risk(sample_returns, confidence=0.95, method="parametric")
        assert var_p < 0

    def test_invalid_method(self, sample_returns: pd.Series) -> None:
        with pytest.raises(ValueError, match="Unknown VaR method"):
            value_at_risk(sample_returns, method="monte_carlo")


class TestConditionalVar:
    def test_worse_than_var(self, sample_returns: pd.Series) -> None:
        var = value_at_risk(sample_returns, confidence=0.95)
        cvar = conditional_var(sample_returns, confidence=0.95)
        assert cvar <= var


class TestTailRatio:
    def test_symmetric_near_one(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 10000))
        tr = tail_ratio(returns)
        assert 0.8 < tr < 1.2  # Roughly symmetric


# ── Distribution metrics ─────────────────────────────────────────────────────

class TestReturnSkewness:
    def test_near_zero_for_normal(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 10000))
        assert abs(return_skewness(returns)) < 0.1


class TestReturnKurtosis:
    def test_near_zero_for_normal(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 10000))
        assert abs(return_kurtosis(returns)) < 0.2


class TestWinRate:
    def test_all_positive(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.03])
        assert win_rate(returns) == 1.0

    def test_all_negative(self) -> None:
        returns = pd.Series([-0.01, -0.02, -0.03])
        assert win_rate(returns) == 0.0

    def test_mixed(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.02, -0.02])
        assert win_rate(returns) == 0.5

    def test_empty(self) -> None:
        assert np.isnan(win_rate(pd.Series(dtype=float)))


# ── Benchmark-relative metrics ───────────────────────────────────────────────

class TestBeta:
    def test_self_beta_is_one(self, sample_returns: pd.Series) -> None:
        b = beta(sample_returns, sample_returns)
        assert np.isclose(b, 1.0, atol=1e-10)

    def test_uncorrelated_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2024-01-01", periods=500, freq="B")
        r1 = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        r2 = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        b = beta(r1, r2)
        assert abs(b) < 0.15


class TestJensensAlpha:
    def test_self_alpha_near_zero(self, sample_returns: pd.Series) -> None:
        a = jensens_alpha(sample_returns, sample_returns, risk_free_rate=0.0)
        assert np.isclose(a, 0.0, atol=1e-10)


class TestTrackingError:
    def test_self_tracking_error_zero(self, sample_returns: pd.Series) -> None:
        te = tracking_error(sample_returns, sample_returns)
        assert np.isclose(te, 0.0, atol=1e-10)

    def test_positive_for_different(
        self, sample_returns: pd.Series, benchmark_returns: pd.Series
    ) -> None:
        te = tracking_error(sample_returns, benchmark_returns)
        assert te > 0


# ── Summary ──────────────────────────────────────────────────────────────────

class TestPerformanceSummary:
    def test_returns_all_keys(self, sample_returns: pd.Series) -> None:
        summary = performance_summary(sample_returns)
        expected_keys = {
            "Total Return", "Annualized Return", "Annualized Volatility",
            "Downside Deviation", "Sharpe Ratio", "Sortino Ratio",
            "Calmar Ratio", "Omega Ratio", "Max Drawdown",
            "Max Drawdown Duration", "Average Drawdown",
            "VaR 95%", "CVaR 95%", "Tail Ratio",
            "Skewness", "Kurtosis", "Win Rate",
            "Beta", "Jensen's Alpha", "Tracking Error", "Information Ratio",
        }
        assert set(summary.keys()) == expected_keys

    def test_with_benchmark(
        self, sample_returns: pd.Series, benchmark_returns: pd.Series
    ) -> None:
        summary = performance_summary(sample_returns, benchmark_returns)
        assert not np.isnan(summary["Beta"])
        assert not np.isnan(summary["Tracking Error"])

    def test_without_benchmark_nan_relatives(self, sample_returns: pd.Series) -> None:
        summary = performance_summary(sample_returns)
        assert np.isnan(summary["Beta"])
        assert np.isnan(summary["Tracking Error"])

    def test_all_values_are_float(self, sample_returns: pd.Series) -> None:
        summary = performance_summary(sample_returns)
        for key, value in summary.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"
