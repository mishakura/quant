# Quantitative Trading & Portfolio Management

A comprehensive collection of quantitative finance strategies, portfolio optimization algorithms, and performance measurement tools for both global and Argentine markets.

## Overview

This repository contains:

- **Factor-Based Strategies**: 9+ quantitative equity screening strategies
- **Portfolio Optimization**: Advanced algorithms (HRP, NCO, minimum correlation, tail risk)
- **Risk Management**: Dynamic hedging and volatility protection strategies
- **Trend Following**: Complete systematic trading framework
- **Performance Analytics**: Portfolio measurement, backtesting, and attribution
- **Argentine Markets**: Integration with BYMA, CME futures, bonds, and local indices
- **Monte Carlo Simulation**: Product and scenario modeling

## Factor Strategies

### Value & Quality
| Strategy | Description | Key Metrics |
|----------|-------------|-------------|
| **QV** (Quantitative Value) | Comprehensive value screening combining franchise power and financial strength | F-Score, ROA, ROC, FCF/Assets, Margin Stability, EBIT/EV |
| **NOA** | Net Operating Assets screening | Operating Assets - Operating Liabilities |
| **CFON** | Cash flow analysis | Cash flow patterns and weights |

### Momentum & Earnings
| Strategy | Description | Rebalance |
|----------|-------------|-----------|
| **Momentum** | Classic price momentum (12-month returns) | Quarterly |
| **Sector Momentum** | SPDR sector ETF rotation | 11-month lookback |
| **SUE** | Standardized Unexpected Earnings | Post-earnings |
| **CAR3** | Cumulative Abnormal Returns (3-day earnings window) | Event-driven |

### Volatility & Risk
| Strategy | Description | Focus |
|----------|-------------|-------|
| **Low Volatility** | Low-vol anomaly screening | Downside protection |
| **Skewness** | Negative skewness portfolio construction | Tail risk mitigation |
| **ATR** | Average True Range signals | Volatility-adjusted positioning |

## Portfolio Optimization

Advanced optimization algorithms using hierarchical clustering and modern portfolio theory:

- **NCO** (Nested Clustered Optimization): Combines hierarchical clustering with optimization across 13 risk measures (MV, CVaR, CDaR, etc.)
- **HRP** (Hierarchical Risk Parity): Dendrogram-based allocation
- **Minimum Correlation**: Decorrelation-focused portfolio construction
- **Tail Risk**: Tail-aware optimization with VaR/CVaR constraints

Risk measures supported: Variance, MAD, CVaR, VaR, Maximum Drawdown, CDaR, EDaR, Ulcer Index, and more.

## Risk Management & Hedging

### Hedging Strategies
- **Dynamic Hedge**: EMA-based tactical allocation (SPX/Gold/Cash)
- **Hedge Vol**: VXX-based volatility hedging during drawdowns
- **Fail Hedge**: Short SPY experiment (archived)

## Trend Following System

Complete systematic trading framework (`/trend`):

1. **Data Pipeline**: Download and clean price data
2. **Indicators**: Technical indicator computation
3. **Signals**: Rule-based signal generation
4. **Execution**: Trade simulation and position management
5. **Analytics**: P&L tracking and performance statistics

Run `update_all.py` to update data, indicators, and signals.

## Performance Measurement

**Portfolio Analytics** (`/pmeasurement`):
- Multi-portfolio tracking and rebalancing
- Risk-adjusted metrics (Sharpe, Sortino, Calmar, Information Ratio)
- Drawdown analysis and visualization
- Beta/Alpha attribution vs SPY
- YTD, quarterly, and multi-period returns
- Argentine portfolio performance (BYMA assets)

## Argentine Market Integration

### BYMA (Bolsas y Mercados Argentinos)
- Excel add-in integration for market data
- Local equity screening and analysis

### CME Data & Instruments
- **Bonares**: Argentine sovereign bonds
- **ON**: Obligaciones Negociables (corporate bonds)
- **Pesos**: Peso-denominated rates and FCIs
- **Tasa Fija**: Fixed rate instruments
- **Indices**: Custom Argentine market indices (CPI-adjusted, etc.)

## Monte Carlo Simulation

Product simulation framework (`/montecarlo`):
- Scenario generation for structured products
- Multi-asset path simulation
- Statistical output analysis

## Main Portfolio Backtest

**`portfolio.py`**: Multi-asset portfolio backtest
- **Assets**: VT (50%), BNDW (20%), ALT (30%)
- **ALT Index**: Equal-weight synthetic from VNQ, REM, IGF, XTN, WOOD, DBC, PSP, QAI
- **Metrics**: Volatility, CAGR, max drawdown, covariance
- **Output**: Returns, cumulative performance, covariance matrix

## Quick Start

```bash
# Install dependencies
pip install pandas numpy yfinance openpyxl scipy riskfolio requests

# Run portfolio backtest
python portfolio.py

# Run a factor strategy (e.g., momentum)
cd strategies/factor/momentum
python momentum.py

# Run portfolio optimization
cd optimization
python nco.py  # NCO optimization

# Update trend following system
cd strategies/trend_following
python update_all.py
```

## Project Structure

```
quant/
├── portfolio.py              # Main portfolio backtest
├── README.md
│
├── strategies/
│   ├── factor/               # Factor-based strategies
│   │   ├── momentum/         # Momentum screening
│   │   ├── low_volatility/   # Low volatility anomaly
│   │   ├── quantitative_value/  # Comprehensive value screening
│   │   ├── skewness/         # Skewness-based allocation
│   │   ├── sue/              # Standardized Unexpected Earnings
│   │   ├── car3/             # 3-day earnings CAR
│   │   ├── noa/              # Net Operating Assets
│   │   ├── atr/              # Average True Range
│   │   └── sector_momentum/  # SPDR sector ETF rotation
│   │
│   ├── trend_following/      # Complete systematic framework
│   │   ├── data.py           # Data download
│   │   ├── indicators.py     # Technical indicators
│   │   ├── signals.py        # Signal generation
│   │   ├── trades.py         # Trade simulation
│   │   └── stats.py          # Performance stats
│   │
│   └── hedging/              # Risk management strategies
│       ├── dynamic_hedge/    # Dynamic hedging (SPX/GC/Cash)
│       ├── hedge_vol/        # VXX volatility hedge
│       └── fail_hedge/       # Short SPY (archived)
│
├── optimization/             # Portfolio optimization
│   ├── nco.py                # Nested Clustered Optimization
│   ├── h.py                  # Hierarchical Risk Parity
│   ├── mincorr.py            # Minimum correlation
│   └── tail.py               # Tail risk optimization
│
├── performance/              # Portfolio analytics
│   ├── performance.py        # Metrics calculation
│   ├── bd.py                 # BYMA data processing
│   ├── rebalance_weights.py  # Rebalancing
│   └── ...                   # Other performance tools
│
├── simulation/               # Simulations
│   └── montecarlo/           # Monte Carlo product simulations
│
└── data/
    ├── market/               # Historical price data (CSVs)
    ├── output/               # Generated reports
    └── argentine/            # Argentine market data
        ├── byma/             # BYMA market data
        ├── cme/              # CME data
        │   ├── bonares/      # Argentine bonds
        │   ├── on/           # Obligaciones Negociables
        │   ├── pesos/        # Peso instruments
        │   └── tasafija/     # Fixed rate instruments
        └── cfon/             # Cash flow analysis
```

## Output Files

Results are saved to various directories:
- `data/output/`: Portfolio backtest results (returns, cumulative, covariance)
- `strategies/factor/*/`: Strategy-specific screening results (*_output.csv)
- `optimization/`: Portfolio weights, dendrograms, statistics
- `strategies/trend_following/`: Daily P&L, trade logs, performance stats
- `performance/`: Drawdown charts, performance reports

## Key Features

- **Multi-Market**: Global equities, bonds, commodities, crypto, Argentine assets
- **Institutional-Grade**: F-Score, NCO, HRP, tail risk optimization
- **Production-Ready**: Rebalancing logic, performance tracking, risk monitoring
- **Research-Focused**: Extensive factor library, backtesting framework
- **Local Integration**: BYMA Excel add-in, CME Argentine data

---

**Disclaimer**: This repository is for research and educational purposes only. Past performance does not guarantee future results. Strategies may involve significant risk and are not suitable for all investors.
