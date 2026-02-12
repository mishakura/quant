"""Tests for quant.utils.validation."""

import numpy as np
import pandas as pd
import pytest

from quant.exceptions import DataError, InsufficientDataError
from quant.utils.validation import (
    check_no_negative_prices,
    validate_returns,
    validate_weights,
)


class TestValidateReturns:
    def test_valid_series(self, sample_returns: pd.Series) -> None:
        result = validate_returns(sample_returns)
        assert result is sample_returns

    def test_valid_dataframe(self, sample_returns_df: pd.DataFrame) -> None:
        result = validate_returns(sample_returns_df)
        assert result is sample_returns_df

    def test_rejects_non_pandas(self) -> None:
        with pytest.raises(DataError, match="Expected pd.Series"):
            validate_returns([0.01, 0.02, -0.01])

    def test_rejects_empty_series(self) -> None:
        with pytest.raises(DataError, match="empty"):
            validate_returns(pd.Series(dtype=float))

    def test_insufficient_observations(self) -> None:
        returns = pd.Series([0.01])
        with pytest.raises(InsufficientDataError):
            validate_returns(returns, min_observations=2)

    def test_rejects_nan_by_default(self) -> None:
        returns = pd.Series([0.01, np.nan, 0.02])
        with pytest.raises(DataError, match="NaN"):
            validate_returns(returns)

    def test_allows_nan_when_permitted(self) -> None:
        returns = pd.Series([0.01, np.nan, 0.02])
        result = validate_returns(returns, allow_nan=True, min_observations=2)
        assert result is returns


class TestValidateWeights:
    def test_valid_weights(self, sample_weights: pd.Series) -> None:
        result = validate_weights(sample_weights)
        assert result is sample_weights

    def test_rejects_non_series(self) -> None:
        with pytest.raises(DataError, match="Expected pd.Series"):
            validate_weights([0.5, 0.5])

    def test_rejects_empty(self) -> None:
        with pytest.raises(DataError, match="empty"):
            validate_weights(pd.Series(dtype=float))

    def test_rejects_nan_weights(self) -> None:
        weights = pd.Series([0.5, np.nan], index=["A", "B"])
        with pytest.raises(DataError, match="NaN"):
            validate_weights(weights)

    def test_rejects_bad_sum(self) -> None:
        weights = pd.Series([0.3, 0.3], index=["A", "B"])
        with pytest.raises(DataError, match="sum to"):
            validate_weights(weights)

    def test_rejects_negative_weights(self) -> None:
        weights = pd.Series([1.5, -0.5], index=["A", "B"])
        with pytest.raises(DataError, match="Negative"):
            validate_weights(weights)

    def test_allows_negative_when_permitted(self) -> None:
        weights = pd.Series([1.5, -0.5], index=["A", "B"])
        result = validate_weights(weights, allow_negative=True)
        assert result is weights


class TestCheckNoNegativePrices:
    def test_valid_prices(self, sample_prices: pd.Series) -> None:
        # Should not raise
        check_no_negative_prices(sample_prices)

    def test_negative_price_raises(self) -> None:
        prices = pd.Series(
            [100, 101, -5, 102],
            index=pd.date_range("2024-01-01", periods=4),
        )
        with pytest.raises(DataError, match="negative"):
            check_no_negative_prices(prices)

    def test_dataframe_negative_raises(self) -> None:
        prices = pd.DataFrame(
            {"A": [100, 101], "B": [50, -1]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        with pytest.raises(DataError, match="negative"):
            check_no_negative_prices(prices)

    def test_valid_dataframe(self, sample_prices_df: pd.DataFrame) -> None:
        check_no_negative_prices(sample_prices_df)
