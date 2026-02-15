"""Shared fixtures for factor tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_factor_prices() -> pd.DataFrame:
    """Price data for 20 assets over 800 business days.

    Assets have different drift and volatility so that factor rankings
    produce meaningful variation.
    """
    rng = np.random.default_rng(42)
    n_assets = 20
    n_days = 800

    dates = pd.bdate_range("2021-01-02", periods=n_days, freq="B")
    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]

    daily_returns = np.column_stack([
        rng.normal(0.0003 + i * 0.00002, 0.01 + i * 0.0005, n_days)
        for i in range(n_assets)
    ])

    prices = 100.0 * np.cumprod(1 + daily_returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def sample_sector_etf_prices() -> pd.DataFrame:
    """Price data for 11 SPDR sector ETFs over 300 business days."""
    rng = np.random.default_rng(99)
    n_days = 300

    etfs = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE"]
    dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")

    daily_returns = np.column_stack([
        rng.normal(0.0002 + (i % 3) * 0.0001, 0.008, n_days)
        for i in range(len(etfs))
    ])

    prices = 100.0 * np.cumprod(1 + daily_returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=etfs)


@pytest.fixture
def short_prices() -> pd.DataFrame:
    """Very short price data (50 days) for testing insufficient data errors."""
    rng = np.random.default_rng(123)
    n_days = 50
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    tickers = ["A", "B", "C"]

    daily_returns = np.column_stack([
        rng.normal(0.0005, 0.015, n_days) for _ in range(3)
    ])

    prices = 100.0 * np.cumprod(1 + daily_returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)
