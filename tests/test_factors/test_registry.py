"""Tests for the factor registry."""

import pytest

from quant.factors.base import Factor
from quant.factors.registry import _FACTOR_REGISTRY, get_factor, list_factors


class TestRegistry:
    """Tests for register_factor / get_factor / list_factors."""

    def test_builtin_factors_registered(self) -> None:
        """All four concrete factors are registered at import time."""
        names = list_factors()
        assert "momentum_12m" in names
        assert "skewness_252d" in names
        assert "low_volatility" in names
        assert "sector_momentum_11m" in names

    def test_get_factor_returns_class(self) -> None:
        cls = get_factor("momentum_12m")
        assert issubclass(cls, Factor)

    def test_get_factor_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            get_factor("nonexistent")

    def test_list_factors_sorted(self) -> None:
        names = list_factors()
        assert names == sorted(names)

    def test_registry_has_at_least_four(self) -> None:
        assert len(_FACTOR_REGISTRY) >= 4
