"""Tests for quant.data.loaders."""

import numpy as np
import pandas as pd
import pytest

from quant.data.loaders import load_returns_from_prices
from quant.exceptions import DataError


class TestLoadReturnsFromPrices:
    def test_arithmetic_returns(self, sample_prices_df: pd.DataFrame) -> None:
        returns = load_returns_from_prices(sample_prices_df, method="arithmetic")
        # First row dropped
        assert len(returns) == len(sample_prices_df) - 1
        # Returns should be roughly centered around zero
        assert returns.mean().abs().max() < 0.01

    def test_log_returns(self, sample_prices_df: pd.DataFrame) -> None:
        returns = load_returns_from_prices(sample_prices_df, method="log")
        assert len(returns) == len(sample_prices_df) - 1
        # Log returns should be close to arithmetic for small values
        arith = load_returns_from_prices(sample_prices_df, method="arithmetic")
        assert np.allclose(returns.values, arith.values, atol=0.01)

    def test_invalid_method_raises(self, sample_prices_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown return method"):
            load_returns_from_prices(sample_prices_df, method="invalid")

    def test_preserves_columns(self, sample_prices_df: pd.DataFrame) -> None:
        returns = load_returns_from_prices(sample_prices_df)
        assert list(returns.columns) == list(sample_prices_df.columns)
