"""Tests for the Factor abstract base class."""

import numpy as np
import pandas as pd
import pytest

from quant.factors.base import Factor, rank_scores


class TestFactorABC:
    """Test that Factor cannot be instantiated directly."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Factor(name="test", lookback=10)  # type: ignore[abstract]

    def test_repr(self) -> None:
        """Concrete subclass repr includes class name and params."""

        class DummyFactor(Factor):
            def compute(self, prices, **kwargs):
                return pd.DataFrame()

            def validate_data(self, prices, **kwargs):
                pass

        f = DummyFactor(name="dummy", lookback=42)
        assert "DummyFactor" in repr(f)
        assert "42" in repr(f)


class TestRankScores:
    """Tests for the rank_scores helper."""

    def test_basic_ranking(self) -> None:
        scores = pd.Series([10, 30, 20, 40])
        ranks = rank_scores(scores)
        # Highest score (40) should have rank 100
        assert np.isclose(ranks.iloc[3], 100.0)
        # Lowest score (10) should have rank 25
        assert np.isclose(ranks.iloc[0], 25.0)

    def test_all_equal(self) -> None:
        scores = pd.Series([5.0, 5.0, 5.0])
        ranks = rank_scores(scores)
        # All equal → all get the same average rank
        # pandas rank(pct=True) averages: (1/3 + 2/3 + 3/3) / 3 = 2/3
        assert np.allclose(ranks, ranks.iloc[0])  # all identical

    def test_single_element(self) -> None:
        scores = pd.Series([42.0])
        ranks = rank_scores(scores)
        assert np.isclose(ranks.iloc[0], 100.0)

    def test_preserves_index(self) -> None:
        scores = pd.Series([1, 3, 2], index=["A", "B", "C"])
        ranks = rank_scores(scores)
        assert list(ranks.index) == ["A", "B", "C"]
