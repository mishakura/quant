"""Tests for MomentumFactor."""

import numpy as np
import pandas as pd
import pytest

from quant.exceptions import DataError, InsufficientDataError
from quant.factors.momentum import MomentumFactor


class TestMomentumFactor:
    """Tests for the 12-month momentum + FIP factor."""

    def test_compute_basic(self, sample_factor_prices: pd.DataFrame) -> None:
        """Factor produces valid output with 20-asset synthetic data."""
        factor = MomentumFactor()
        result = factor.compute(sample_factor_prices)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"ticker", "momentum", "fip", "score", "rank"}
        assert len(result) > 0
        # All scores should be finite
        assert result["score"].notna().all()

    def test_output_columns(self, sample_factor_prices: pd.DataFrame) -> None:
        """Output contains exactly the expected columns in order."""
        factor = MomentumFactor()
        result = factor.compute(sample_factor_prices)
        assert list(result.columns) == ["ticker", "momentum", "fip", "score", "rank"]

    def test_score_is_negative_fip(self, sample_factor_prices: pd.DataFrame) -> None:
        """Score should equal -FIP."""
        factor = MomentumFactor()
        result = factor.compute(sample_factor_prices)
        np.testing.assert_allclose(result["score"].values, -result["fip"].values)

    def test_top_pct_selection(self, sample_factor_prices: pd.DataFrame) -> None:
        """Output count respects top_momentum_pct × top_fip_pct."""
        factor = MomentumFactor(top_momentum_pct=0.50, top_fip_pct=1.0)
        result = factor.compute(sample_factor_prices)
        # With 20 assets, 50% momentum = ~10, 100% FIP = all
        assert len(result) <= 10

    def test_insufficient_data_raises(self, short_prices: pd.DataFrame) -> None:
        """Prices shorter than lookback raise InsufficientDataError."""
        factor = MomentumFactor(lookback=280)
        with pytest.raises(InsufficientDataError):
            factor.compute(short_prices)

    def test_all_nan_column_raises(self) -> None:
        """All-NaN ticker raises DataError."""
        dates = pd.bdate_range("2023-01-02", periods=300, freq="B")
        prices = pd.DataFrame({
            "A": np.random.default_rng(1).normal(0.001, 0.01, 300).cumsum() + 100,
            "B": np.nan,
        }, index=dates)
        factor = MomentumFactor()
        with pytest.raises(DataError):
            factor.compute(prices)

    def test_custom_lookback(self, sample_factor_prices: pd.DataFrame) -> None:
        """Factor works with a shorter lookback."""
        factor = MomentumFactor(lookback=200)
        result = factor.compute(sample_factor_prices)
        assert len(result) > 0

    def test_rank_range(self, sample_factor_prices: pd.DataFrame) -> None:
        """Ranks should be in [0, 100]."""
        factor = MomentumFactor()
        result = factor.compute(sample_factor_prices)
        assert (result["rank"] >= 0).all()
        assert (result["rank"] <= 100).all()
