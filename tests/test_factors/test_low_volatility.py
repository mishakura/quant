"""Tests for LowVolatilityFactor."""

import numpy as np
import pandas as pd
import pytest

from quant.exceptions import InsufficientDataError
from quant.factors.low_volatility import LowVolatilityFactor


class TestLowVolatilityFactor:
    """Tests for the low-vol anomaly factor."""

    def test_compute_basic(self, sample_factor_prices: pd.DataFrame) -> None:
        """Factor produces valid output."""
        factor = LowVolatilityFactor()
        result = factor.compute(sample_factor_prices)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"ticker", "volatility", "score", "rank"}
        assert len(result) > 0

    def test_score_is_negative_vol(self, sample_factor_prices: pd.DataFrame) -> None:
        """Score = -volatility."""
        factor = LowVolatilityFactor()
        result = factor.compute(sample_factor_prices)
        np.testing.assert_allclose(result["score"].values, -result["volatility"].values)

    def test_lower_vol_higher_rank(self) -> None:
        """Asset with lower volatility gets a higher rank."""
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2021-01-02", periods=800, freq="B")

        # LOW_VOL: low sigma → should rank higher
        low_vol = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.005, 800))
        # HIGH_VOL: high sigma → should rank lower
        high_vol = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.03, 800))

        prices = pd.DataFrame({"LOW_VOL": low_vol, "HIGH_VOL": high_vol}, index=dates)
        factor = LowVolatilityFactor()
        result = factor.compute(prices)

        low_rank = result.loc[result["ticker"] == "LOW_VOL", "rank"].values[0]
        high_rank = result.loc[result["ticker"] == "HIGH_VOL", "rank"].values[0]
        assert low_rank > high_rank

    def test_insufficient_data(self, short_prices: pd.DataFrame) -> None:
        """Short data raises InsufficientDataError."""
        factor = LowVolatilityFactor(lookback=252)
        with pytest.raises(InsufficientDataError):
            factor.compute(short_prices)

    def test_volatility_positive(self, sample_factor_prices: pd.DataFrame) -> None:
        """Volatility values should all be positive."""
        factor = LowVolatilityFactor()
        result = factor.compute(sample_factor_prices)
        assert (result["volatility"] > 0).all()
