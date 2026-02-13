"""Tests for SkewnessFactor."""

import numpy as np
import pandas as pd
import pytest

from quant.exceptions import InsufficientDataError
from quant.factors.skewness import SkewnessFactor


class TestSkewnessFactor:
    """Tests for the return skewness factor."""

    def test_compute_basic(self, sample_factor_prices: pd.DataFrame) -> None:
        """Factor produces valid output."""
        factor = SkewnessFactor()
        result = factor.compute(sample_factor_prices)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"ticker", "skewness", "score", "rank"}
        assert len(result) > 0

    def test_score_is_negative_skewness(self, sample_factor_prices: pd.DataFrame) -> None:
        """Score = -skewness."""
        factor = SkewnessFactor()
        result = factor.compute(sample_factor_prices)
        np.testing.assert_allclose(result["score"].values, -result["skewness"].values)

    def test_insufficient_data(self, short_prices: pd.DataFrame) -> None:
        """Short data raises InsufficientDataError."""
        factor = SkewnessFactor(lookback=252)
        with pytest.raises(InsufficientDataError):
            factor.compute(short_prices)

    def test_output_sorted_by_rank(self, sample_factor_prices: pd.DataFrame) -> None:
        """Output is sorted by rank descending."""
        factor = SkewnessFactor()
        result = factor.compute(sample_factor_prices)
        assert result["rank"].is_monotonic_decreasing or len(result) <= 1

    def test_rank_range(self, sample_factor_prices: pd.DataFrame) -> None:
        """Ranks in [0, 100]."""
        factor = SkewnessFactor()
        result = factor.compute(sample_factor_prices)
        assert (result["rank"] >= 0).all()
        assert (result["rank"] <= 100).all()


class TestComputeRankWeights:
    """Tests for the Asness/Koijen rank-weight method."""

    def test_positive_weights_sum_to_one(self, sample_factor_prices: pd.DataFrame) -> None:
        factor = SkewnessFactor()
        scores = factor.compute(sample_factor_prices)
        weights = SkewnessFactor.compute_rank_weights(scores)
        pos = weights[weights > 0]
        if len(pos) > 0:
            assert np.isclose(pos.sum(), 1.0, atol=1e-10)

    def test_negative_weights_sum_to_minus_one(
        self, sample_factor_prices: pd.DataFrame,
    ) -> None:
        factor = SkewnessFactor()
        scores = factor.compute(sample_factor_prices)
        weights = SkewnessFactor.compute_rank_weights(scores)
        neg = weights[weights < 0]
        if len(neg) > 0:
            assert np.isclose(neg.sum(), -1.0, atol=1e-10)

    def test_single_asset_returns_zero(self) -> None:
        """Single asset → weight = 0 (need at least 2)."""
        scores = pd.DataFrame({"ticker": ["A"], "skewness": [0.5]})
        weights = SkewnessFactor.compute_rank_weights(scores)
        assert np.isclose(weights.iloc[0], 0.0)

    def test_two_assets(self) -> None:
        """Two assets: one long (+1), one short (-1)."""
        scores = pd.DataFrame({
            "ticker": ["NEG_SKEW", "POS_SKEW"],
            "skewness": [-2.0, 2.0],
        })
        weights = SkewnessFactor.compute_rank_weights(scores)
        # NEG_SKEW has more negative skew → higher rank → positive weight
        assert weights["NEG_SKEW"] > 0
        assert weights["POS_SKEW"] < 0
