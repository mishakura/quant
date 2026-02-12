"""Shared test fixtures for the quant test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_returns() -> pd.Series:
    """Daily return series (500 days) with realistic equity-like properties.

    Mean ~8% annualized, vol ~16% annualized.
    """
    rng = np.random.default_rng(42)
    n = 500
    daily_mu = 0.08 / 252
    daily_sigma = 0.16 / np.sqrt(252)
    returns = pd.Series(
        rng.normal(daily_mu, daily_sigma, n),
        index=pd.bdate_range("2022-01-03", periods=n, freq="B"),
        name="TEST_ASSET",
    )
    return returns


@pytest.fixture
def sample_prices(sample_returns: pd.Series) -> pd.Series:
    """Price series derived from sample_returns, starting at 100."""
    prices = 100.0 * (1 + sample_returns).cumprod()
    prices.name = "TEST_ASSET"
    return prices


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Multi-asset daily return DataFrame (500 days, 5 assets).

    Assets have different risk/return profiles and moderate correlation.
    """
    rng = np.random.default_rng(123)
    n = 500
    assets = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]

    # Generate correlated returns via Cholesky decomposition
    # Correlation matrix with moderate positive correlation
    corr = np.array([
        [1.0, 0.6, 0.3, 0.2, 0.1],
        [0.6, 1.0, 0.4, 0.3, 0.2],
        [0.3, 0.4, 1.0, 0.5, 0.3],
        [0.2, 0.3, 0.5, 1.0, 0.4],
        [0.1, 0.2, 0.3, 0.4, 1.0],
    ])
    # Annualized vols: 16%, 20%, 12%, 25%, 18%
    vols = np.array([0.16, 0.20, 0.12, 0.25, 0.18]) / np.sqrt(252)
    cov = np.outer(vols, vols) * corr
    chol = np.linalg.cholesky(cov)

    # Annualized means: 8%, 10%, 6%, 12%, 7%
    means = np.array([0.08, 0.10, 0.06, 0.12, 0.07]) / 252

    raw = rng.standard_normal((n, 5))
    returns_data = raw @ chol.T + means

    return pd.DataFrame(
        returns_data,
        index=pd.bdate_range("2022-01-03", periods=n, freq="B"),
        columns=assets,
    )


@pytest.fixture
def sample_prices_df(sample_returns_df: pd.DataFrame) -> pd.DataFrame:
    """Multi-asset price DataFrame derived from sample_returns_df, starting at 100."""
    return 100.0 * (1 + sample_returns_df).cumprod()


@pytest.fixture
def sample_weights() -> pd.Series:
    """Equal-weight portfolio for 5 assets."""
    return pd.Series(
        [0.2, 0.2, 0.2, 0.2, 0.2],
        index=["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"],
        name="weights",
    )


@pytest.fixture
def benchmark_returns() -> pd.Series:
    """Benchmark (market) return series for 500 days."""
    rng = np.random.default_rng(99)
    n = 500
    daily_mu = 0.09 / 252
    daily_sigma = 0.15 / np.sqrt(252)
    return pd.Series(
        rng.normal(daily_mu, daily_sigma, n),
        index=pd.bdate_range("2022-01-03", periods=n, freq="B"),
        name="BENCHMARK",
    )
