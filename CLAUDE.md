# CLAUDE.md — Quant Project Blueprint

> This document defines the **target architecture, standards, and conventions** for this quantitative finance project.
> It serves as the authoritative reference for all refactoring and new development.
> The goal is to bring this project to **institutional-grade quality** following professional quant practices.

---

## Project Identity

- **Name**: `quant` — Quantitative Finance Research & Portfolio Management Platform
- **Language**: Python 3.11+
- **Paradigm**: Object-oriented core with functional utilities. NumPy/Pandas vectorized computation everywhere — no loops for numerical work.
- **Domain**: Factor investing, portfolio optimization, systematic trading, risk management, Monte Carlo simulation, Argentine fixed income analytics.

---

## Target Directory Structure

```
quant/
├── pyproject.toml                  # Single source of truth for project config
├── CLAUDE.md
├── README.md
├── .env.example                    # Template for environment variables
├── .gitignore
│
├── src/
│   └── quant/                      # Installable package (pip install -e .)
│       ├── __init__.py
│       ├── config.py               # Centralized configuration (paths, constants, params)
│       │
│       ├── data/                   # Data layer — download, clean, store, serve
│       │   ├── __init__.py
│       │   ├── providers/          # Data source adapters (yfinance, BYMA, CME)
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # Abstract DataProvider interface
│       │   │   ├── yfinance.py
│       │   │   ├── byma.py
│       │   │   └── cme.py
│       │   ├── loaders.py          # Load from local cache (CSV/Parquet)
│       │   ├── cleaners.py         # Data validation, NaN handling, alignment
│       │   └── universe.py         # Asset universe definitions and filtering
│       │
│       ├── factors/                # Factor research and signal generation
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract Factor class with standard interface
│       │   ├── momentum.py         # Price momentum (cross-sectional & time-series)
│       │   ├── value.py            # Quantitative value (F-Score, EBIT/EV, FCF/Assets)
│       │   ├── low_volatility.py   # Low-vol anomaly
│       │   ├── skewness.py         # Return skewness factor
│       │   ├── quality.py          # ROA, ROC, margin stability
│       │   ├── earnings.py         # SUE, CAR3, post-earnings drift
│       │   ├── sector_rotation.py  # Sector/ETF momentum rotation
│       │   └── registry.py         # Factor registry for discovery and composition
│       │
│       ├── signals/                # Signal generation and combination
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract Signal class
│       │   ├── technical.py        # Technical indicators (ATR, moving averages, breakouts)
│       │   ├── fundamental.py      # Fundamental signals from factor scores
│       │   ├── composite.py        # Multi-signal combination and ranking
│       │   └── filters.py          # Universe filters (liquidity, market cap, exchange)
│       │
│       ├── portfolio/              # Portfolio construction and optimization
│       │   ├── __init__.py
│       │   ├── optimizer.py        # Base optimizer interface
│       │   ├── mean_variance.py    # Classical MVO with shrinkage estimators
│       │   ├── hierarchical.py     # HRP, HERC (hierarchical risk parity)
│       │   ├── nco.py              # Nested Clustered Optimization
│       │   ├── min_correlation.py  # Minimum correlation algorithm
│       │   ├── risk_budgeting.py   # Risk parity and risk budgeting
│       │   ├── constraints.py      # Weight constraints, turnover limits, sector bounds
│       │   └── rebalance.py        # Rebalancing logic (calendar, threshold, hybrid)
│       │
│       ├── risk/                   # Risk measurement and management
│       │   ├── __init__.py
│       │   ├── measures.py         # VaR, CVaR, CDaR, MDD, Ulcer Index, etc.
│       │   ├── factor_risk.py      # Factor exposure and attribution
│       │   ├── tail_risk.py        # Tail risk analysis, EVT
│       │   ├── hedging.py          # Dynamic hedging strategies
│       │   └── stress.py           # Stress testing and scenario analysis
│       │
│       ├── backtesting/            # Strategy backtesting engine
│       │   ├── __init__.py
│       │   ├── engine.py           # Event-driven or vectorized backtest loop
│       │   ├── portfolio_state.py  # Track positions, cash, NAV over time
│       │   ├── execution.py        # Simulated execution (slippage, commissions, market impact)
│       │   ├── calendar.py         # Trading calendar and rebalance schedule
│       │   └── results.py          # Backtest results container with standard metrics
│       │
│       ├── simulation/             # Monte Carlo and scenario simulation
│       │   ├── __init__.py
│       │   ├── montecarlo.py       # Path generation (GBM, fat-tailed, correlated)
│       │   ├── distributions.py    # Distribution fitting and parameter estimation
│       │   ├── scenarios.py        # Scenario definition and management
│       │   └── cashflow.py         # Cashflow modeling with inflation adjustment
│       │
│       ├── strategies/             # Complete trading strategy implementations
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract Strategy class
│       │   ├── factor_strategy.py  # Long/short factor portfolio strategy
│       │   ├── trend_following.py  # Systematic trend following
│       │   ├── hedging_overlay.py  # Hedging overlay strategies
│       │   └── registry.py         # Strategy registry
│       │
│       ├── analytics/              # Performance analytics and reporting
│       │   ├── __init__.py
│       │   ├── performance.py      # Returns, Sharpe, Sortino, Calmar, Omega, etc.
│       │   ├── attribution.py      # Performance attribution (Brinson, factor-based)
│       │   ├── benchmark.py        # Benchmark comparison (alpha, beta, tracking error, IR)
│       │   ├── drawdown.py         # Drawdown analysis (depth, duration, recovery)
│       │   └── report.py           # Report generation (HTML, PDF, Excel)
│       │
│       ├── fixed_income/           # Argentine fixed income module
│       │   ├── __init__.py
│       │   ├── bonds.py            # Bond pricing, duration, convexity, spreads
│       │   ├── cashflows.py        # Cashflow construction and NPV
│       │   ├── curves.py           # Yield curve construction and interpolation
│       │   ├── inflation.py        # CPI-linked instrument modeling
│       │   └── instruments/        # Instrument definitions
│       │       ├── __init__.py
│       │       ├── bonares.py      # Sovereign bonds (Bonares, Globales)
│       │       ├── on.py           # Obligaciones Negociables (corporates)
│       │       ├── letras.py       # Short-term bills (Lecap, Lecer, Lede)
│       │       └── tasa_fija.py    # Fixed rate instruments
│       │
│       └── utils/                  # Shared utilities
│           ├── __init__.py
│           ├── dates.py            # Date utilities, business day calendar
│           ├── io.py               # Read/write CSV, Parquet, Excel
│           ├── logging.py          # Structured logging configuration
│           ├── validation.py       # Data validation decorators and helpers
│           └── constants.py        # Trading days (252), risk-free rate, etc.
│
├── notebooks/                      # Jupyter notebooks for research and exploration
│   ├── research/                   # Factor research, signal testing
│   ├── backtests/                  # Strategy backtest analysis
│   └── reports/                    # Client/internal reports
│
├── scripts/                        # CLI entry points and automation
│   ├── run_backtest.py             # Run a backtest from config
│   ├── update_data.py              # Download/refresh market data
│   ├── generate_report.py          # Generate performance reports
│   ├── run_optimization.py         # Run portfolio optimization
│   └── run_simulation.py           # Run Monte Carlo simulations
│
├── tests/                          # Test suite (mirrors src/quant structure)
│   ├── conftest.py                 # Shared fixtures (sample returns, prices, portfolios)
│   ├── test_factors/
│   ├── test_portfolio/
│   ├── test_risk/
│   ├── test_backtesting/
│   ├── test_analytics/
│   ├── test_simulation/
│   └── test_fixed_income/
│
├── data/                           # Data storage (gitignored except samples)
│   ├── raw/                        # Downloaded raw data (never modified)
│   ├── processed/                  # Cleaned, aligned, ready-to-use data
│   ├── cache/                      # Intermediate computation cache
│   └── samples/                    # Small sample datasets for tests (tracked in git)
│
├── configs/                        # Strategy and portfolio configuration files
│   ├── universes/                  # Asset universe definitions (YAML)
│   ├── strategies/                 # Strategy parameter configs (YAML)
│   ├── portfolios/                 # Portfolio construction configs (YAML)
│   └── simulations/                # Monte Carlo simulation configs (YAML)
│
└── output/                         # Generated outputs (gitignored)
    ├── backtests/
    ├── reports/
    ├── optimizations/
    └── simulations/
```

---

## Python Standards & Conventions

### Project Setup

```toml
# pyproject.toml — use modern Python packaging
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "quant"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "yfinance>=0.2",
    "riskfolio-lib>=6.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "openpyxl>=3.1",
    "pyyaml>=6.0",
    "pyarrow>=14.0",       # Parquet support
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov", "ruff", "mypy", "pre-commit"]
notebooks = ["jupyterlab", "ipywidgets"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "NPY", "PD"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: marks tests that download data or run long simulations",
    "integration: marks tests requiring external services",
]
```

### Type Hints — Mandatory

Every function must have complete type annotations. Use modern Python 3.11+ syntax:

```python
import numpy as np
import pandas as pd
from numpy.typing import NDArray

def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio."""
    excess = returns - risk_free_rate / annualization_factor
    rolling_mean = excess.rolling(window).mean() * annualization_factor
    rolling_std = excess.rolling(window).std() * np.sqrt(annualization_factor)
    return rolling_mean / rolling_std
```

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Modules | `snake_case.py` | `trend_following.py` |
| Classes | `PascalCase` | `MomentumFactor`, `HRPOptimizer` |
| Functions/methods | `snake_case` | `compute_max_drawdown()` |
| Constants | `UPPER_SNAKE_CASE` | `TRADING_DAYS_PER_YEAR = 252` |
| Private | `_leading_underscore` | `_validate_weights()` |
| Type aliases | `PascalCase` | `Returns = pd.Series` |
| Config keys (YAML) | `snake_case` | `lookback_window: 252` |

### Docstrings — NumPy Style

```python
def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "hrp",
    risk_measure: str = "mv",
    constraints: dict[str, float] | None = None,
) -> pd.Series:
    """Optimize portfolio weights using the specified method.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return matrix, shape (T, N). Each column is an asset.
    method : str
        Optimization method. One of 'hrp', 'herc', 'nco', 'mvo', 'min_corr'.
    risk_measure : str
        Risk measure for optimization. One of 'mv', 'cvar', 'cdar', 'mdd'.
    constraints : dict[str, float] | None
        Optional weight constraints. Keys: 'max_weight', 'min_weight', 'max_turnover'.

    Returns
    -------
    pd.Series
        Optimal portfolio weights indexed by asset name. Sums to 1.0.

    Raises
    ------
    ValueError
        If method or risk_measure is not supported.
    OptimizationError
        If the optimizer fails to converge.

    Examples
    --------
    >>> weights = optimize_portfolio(returns, method="hrp", risk_measure="cvar")
    >>> assert np.isclose(weights.sum(), 1.0)
    """
```

### Design Patterns

#### 1. Abstract Base Classes for Extensibility

```python
from abc import ABC, abstractmethod

class Factor(ABC):
    """Base class for all factor implementations."""

    def __init__(self, name: str, lookback: int, universe: Universe) -> None:
        self.name = name
        self.lookback = lookback
        self.universe = universe

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute factor scores for all assets. Returns scores indexed by ticker."""
        ...

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that input data meets factor requirements."""
        ...
```

#### 2. Strategy Pattern for Portfolio Optimization

Each optimizer implements a common interface. The backtesting engine is agnostic to which optimizer runs.

#### 3. Registry Pattern for Discoverability

Factors, strategies, and optimizers register themselves so that scripts and configs can reference them by name.

#### 4. Configuration via YAML — Not Hardcoded

```yaml
# configs/strategies/momentum.yaml
strategy:
  name: cross_sectional_momentum
  factor: momentum
  lookback: 252
  holding_period: 63  # quarterly
  universe: sp500
  long_count: 50
  short_count: 50
  rebalance: quarterly
  filters:
    min_market_cap: 1e9
    min_avg_volume: 1e6
    exchanges: ["NYSE", "NASDAQ"]
```

#### 5. Data Pipeline: Raw → Processed → Analysis

Never modify raw data. Always maintain a clear pipeline:
1. **Download** → `data/raw/`
2. **Clean & validate** → `data/processed/`
3. **Compute** → use processed data in memory
4. **Output** → `output/`

---

## Quantitative Finance Standards

### Return Computation

- Use **log returns** for time-series aggregation and statistical testing.
- Use **arithmetic returns** for cross-sectional analysis and portfolio P&L.
- Always specify which return type a function expects via parameter or naming.
- Annualize using `252` trading days (configurable per market).

### Risk Measures — Implement All Standard Metrics

| Measure | Formula Context |
|---|---|
| Volatility | Annualized std dev of returns |
| Sharpe Ratio | (mean excess return) / volatility |
| Sortino Ratio | (mean excess return) / downside deviation |
| Calmar Ratio | Annualized return / max drawdown |
| Omega Ratio | Probability-weighted gains / losses above threshold |
| Max Drawdown | Maximum peak-to-trough decline |
| VaR (parametric & historical) | Value at Risk at 95%/99% confidence |
| CVaR / Expected Shortfall | Mean loss beyond VaR |
| CDaR | Conditional Drawdown at Risk |
| Ulcer Index | RMS of drawdown percentage |
| Skewness & Kurtosis | Higher moments of return distribution |
| Beta & Jensen's Alpha | CAPM regression vs benchmark |
| Tracking Error | Std dev of active returns |
| Information Ratio | Mean active return / tracking error |

### Backtesting Rules

1. **No look-ahead bias**: Signals at time `t` use only data available at `t-1` or earlier.
2. **Transaction costs**: Always model slippage and commissions. Default: 10bps round-trip.
3. **Survivorship bias**: Use point-in-time universe composition when available.
4. **Rebalance lag**: Apply at least 1-day execution lag.
5. **Capacity constraints**: Flag strategies with excessive turnover or illiquid holdings.

### Data Quality

- Validate all downloaded data: check for gaps, duplicates, negative prices, extreme outliers.
- Handle corporate actions (splits, dividends) via adjusted prices.
- Log all data quality issues — never silently drop or fill data.
- Use `NaN` to represent missing data. Never use `0` as a substitute for missing prices or returns.

---

## Error Handling

- Use custom exception hierarchy rooted in a `QuantError` base class.
- Raise `DataError` for data quality issues, `OptimizationError` for solver failures, `ConfigError` for invalid parameters.
- Never use bare `except:`. Always catch specific exceptions.
- Log errors with full context (asset, date, parameters) using structured logging.

```python
class QuantError(Exception):
    """Base exception for all quant module errors."""

class DataError(QuantError):
    """Raised when data validation fails."""

class OptimizationError(QuantError):
    """Raised when portfolio optimization fails to converge."""

class ConfigError(QuantError):
    """Raised for invalid configuration parameters."""
```

---

## Testing Standards

- **Unit tests** for every public function — test edge cases, NaN handling, empty inputs.
- **Property-based tests** for mathematical invariants (e.g., portfolio weights sum to 1, Sharpe ratio is scale-invariant).
- **Regression tests** with known-good outputs for strategies and optimizers.
- **Integration tests** (marked `@pytest.mark.slow`) for full pipeline runs.
- Use `conftest.py` fixtures for sample return series, price data, and portfolio objects.
- Target: **>80% code coverage** on `src/quant/`.

---

## Data Storage Conventions

| Data Type | Format | Location |
|---|---|---|
| Market prices (time series) | Parquet (columnar, compressed) | `data/raw/` or `data/processed/` |
| Strategy configs | YAML | `configs/` |
| Backtest results | Parquet | `output/backtests/` |
| Reports | HTML or Excel | `output/reports/` |
| Optimization weights | CSV or Parquet | `output/optimizations/` |
| Simulation outputs | Parquet | `output/simulations/` |
| Interactive research | Jupyter notebooks | `notebooks/` |
| Small sample data for tests | CSV | `data/samples/` |

**Prefer Parquet over CSV** for all numerical data — faster I/O, smaller files, preserves dtypes.
**Prefer YAML over Excel** for configuration — version-controllable, diffable, no binary bloat.

---

## Git & Version Control

- `.gitignore` must exclude: `data/raw/`, `data/processed/`, `data/cache/`, `output/`, `*.xlsx` (except samples), `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `.env`.
- Track in git: `src/`, `tests/`, `configs/`, `scripts/`, `notebooks/`, `data/samples/`, `pyproject.toml`.
- Commit messages: imperative mood, concise. E.g., `Add HRP optimizer with CVaR risk measure`.
- Branches: `feature/<name>`, `fix/<name>`, `refactor/<name>`.

---

## Logging & Observability

- Use Python's `logging` module with structured output.
- Log levels: `DEBUG` for computation details, `INFO` for pipeline progress, `WARNING` for data quality issues, `ERROR` for failures.
- Every script entry point configures logging before any computation.
- Include timestamps, module name, and relevant context (ticker, date range) in log messages.

---

## Language Policy

- All code, docstrings, comments, variable names, and commit messages: **English**.
- User-facing reports may be bilingual (English/Spanish) based on configuration.
- Configuration keys and YAML files: **English**.

---

## Dependency Policy

Core dependencies (always available):
- `numpy`, `pandas`, `scipy` — numerical computing
- `riskfolio-lib` — portfolio optimization
- `yfinance` — market data
- `matplotlib`, `seaborn` — visualization
- `pyarrow` — Parquet I/O
- `pyyaml` — configuration

Development dependencies:
- `pytest`, `pytest-cov` — testing
- `ruff` — linting and formatting
- `mypy` — static type checking
- `pre-commit` — git hooks

**Do not add dependencies without justification.** Prefer stdlib and existing deps over new packages.

---

## Refactoring Progress

### Phase 1 — Package scaffolding, utilities, data layer, analytics ✅

Created `src/quant/` as an installable package with:

| Module | Status | Contents |
|---|---|---|
| `pyproject.toml` | ✅ Done | Hatchling build, all deps, pytest/ruff/mypy config |
| `utils/constants.py` | ✅ Done | `TRADING_DAYS_PER_YEAR`, `DEFAULT_RISK_FREE_RATE`, `RISK_MEASURES`, `QUARTER_END_DATES` |
| `utils/dates.py` | ✅ Done | `is_quarter_end()`, `last_quarter_end()`, `infer_periods_per_year()`, `get_rebalance_dates()` |
| `utils/validation.py` | ✅ Done | `validate_returns()`, `validate_weights()`, `check_no_negative_prices()` |
| `utils/logging.py` | ✅ Done | Structured logging config with `get_logger()` |
| `utils/io.py` | ✅ Done | CSV/Parquet/Excel read-write helpers |
| `data/loaders.py` | ✅ Done | `load_price_csv()`, `load_returns_from_prices()`, `load_weights_excel()` |
| `data/cleaners.py` | ✅ Done | `find_common_start_date()`, `trim_to_common_history()`, `align_to_common_dates()`, `validate_price_data()` |
| `data/providers/base.py` | ✅ Done | Abstract `DataProvider` interface |
| `data/providers/yfinance.py` | ✅ Done | `YFinanceProvider` with `fetch_prices()` and `fetch_ohlcv()` |
| `analytics/performance.py` | ✅ Done | 17 metric functions + `performance_summary()` (Sharpe, Sortino, Calmar, Omega, VaR, CVaR, beta, alpha, tracking error, IR, drawdown, skewness, kurtosis, win rate) |
| `exceptions.py` | ✅ Done | `QuantError` → `DataError`, `OptimizationError`, `ConfigError`, `InsufficientDataError` |
| `config.py` | ✅ Done | Path constants (`PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`, etc.) + `ensure_directories()` |
| `tests/` | ✅ Done | 106 tests across 6 test files (conftest fixtures, analytics, data, utils) |

**No existing files modified.** Phase 1 was purely additive.

### Phase 2 — Migrate 6 highest-duplication files to use `quant.*` ✅

Replaced ~200 lines of copy-pasted metric calculations with imports from `quant.*`:

| File | What changed |
|---|---|
| `optimization/nco.py` | 20-line inline stats block → `performance_summary()`, local `RISK_MEASURES` → constant, `rf = 0.03` → `DEFAULT_RISK_FREE_RATE`, `252` → `TRADING_DAYS_PER_YEAR` |
| `optimization/mincorr.py` | `summary_stats()` → `annualized_return()` + `sharpe_ratio()`, `returns_from_prices()` → `load_returns_from_prices()`, removed unused `scipy.optimize` import |
| `optimization/h.py` | Created `_compute_stats()` helper wrapping `performance_summary()`, replaced 3 duplicated stats blocks (~200 lines → ~30), removed `scipy.stats` import |
| `strategies/trend_following/stats.py` | `scipy.stats.skew/kurtosis` → `return_skewness()`/`return_kurtosis()`, `yf.download` → `YFinanceProvider`, `252` → `TRADING_DAYS_PER_YEAR` |
| `strategies/hedging/dynamic_hedge/dhedging.py` | Removed 3 local functions (`infer_periods_per_year`, `max_drawdown`, `summarize`) → quant imports, removed `import math` |
| `performance/performance.py` | `is_quarter_end()` + search logic → `last_quarter_end()`, `risk_free_annual = 0.05` → `DEFAULT_RISK_FREE_RATE`, inline metrics → `sharpe_ratio()`, `beta()`, `tracking_error()`, etc. |

**Net result:** 244 insertions, 460 deletions (−216 lines). All risk-free rates standardized to `DEFAULT_RISK_FREE_RATE = 0.05`.

### Phase 3+ — Remaining work (not started)

| # | Phase | Scope | Notes |
|---|---|---|---|
| 6 | **Refactor factors** | Extract momentum, value, skewness, quality, earnings factors into `Factor` ABC classes under `src/quant/factors/` | Existing code in `strategies/factor/` — needs class hierarchy |
| 7 | **Refactor optimizers** | Unify `optimization/nco.py`, `h.py`, `mincorr.py` under `src/quant/portfolio/` with common `Optimizer` interface | Extract Riskfolio wrappers from scripts into reusable classes |
| 8 | **Build backtesting engine** | Consolidate `simulation.py`, `trades.py`, `signals.py` into `src/quant/backtesting/` | Event-driven or vectorized backtest loop |
| 9 | **Migrate strategies** | Wrap trend-following (`dhedging.py`) and hedging into `Strategy` ABC implementations | Keep EMA logic, add proper interface |
| 10 | **Migrate fixed income** | Structure Argentine bonds, corporates, rates under `src/quant/fixed_income/` | `data/argentine/` has raw data; needs pricing models |
| 11 | **Migrate Monte Carlo** | Clean up `simulation/montecarlo/` with proper distribution fitting | GBM, fat-tailed, correlated path generation |
| 12 | **Expand tests** | Add tests for factors, optimizers, backtesting, strategies | Target >80% coverage on `src/quant/` |
| 13 | **Add configs** | Extract hardcoded asset lists, date ranges, rf rates into YAML | `configs/universes/`, `configs/strategies/`, `configs/portfolios/` |
| 14 | **Add scripts** | CLI entry points: `run_backtest.py`, `update_data.py`, `run_optimization.py` | Use `argparse` + YAML config loading |
| 15 | **Convert data to Parquet** | Migrate CSV/Excel data files to Parquet where appropriate | Faster I/O, smaller files, preserves dtypes |

---

## Commands

```bash
# Install in development mode
pip install -e ".[dev,notebooks]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=quant --cov-report=html

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/quant/

# Run a specific script
python scripts/run_backtest.py --config configs/strategies/momentum.yaml
python scripts/update_data.py --universe sp500
python scripts/run_optimization.py --config configs/portfolios/hrp.yaml
```
