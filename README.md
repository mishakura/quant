# Quant Trading Strategies

A collection of quantitative trading strategies, portfolio optimization tools, and backtesting frameworks for global and Argentine markets.

## Overview

This repository contains various quantitative finance strategies and tools including:

- **Portfolio Backtesting**: Multi-asset portfolio analysis with risk metrics (volatility, drawdown, CAGR)
- **Factor Strategies**: Momentum, low volatility, ATR, skewness, and more
- **Optimization**: Portfolio optimization algorithms (minimum correlation, NCO, hierarchical)
- **Risk Management**: Dynamic hedging, volatility hedging strategies
- **Monte Carlo Simulations**: Product simulations and scenario analysis
- **Market Data**: Integration with BYMA (Argentine markets) and global ETFs via yfinance

## Key Strategies

| Strategy | Description | Location |
|----------|-------------|----------|
| **Momentum** | Momentum-based stock selection | `/momentum` |
| **Low Volatility** | Low-vol anomaly strategy | `/low` |
| **ATR** | Average True Range-based signals | `/atr` |
| **Hedging** | Dynamic portfolio hedging | `/dynamic_hedge`, `/hedge_vol` |
| **Portfolio Backtest** | Multi-asset portfolio analysis | `portfolio.py` |

## Markets Covered

- **Global**: US equities, bonds, commodities (via yfinance)
- **Argentina**: BYMA stocks, CME futures, Argentine bonds and rates

## Quick Start

```bash
# Install dependencies
pip install pandas numpy yfinance openpyxl

# Run portfolio backtest
python portfolio.py
```

## Structure

```
quant/
├── portfolio.py          # Main portfolio backtesting script
├── momentum/             # Momentum strategy
├── low/                  # Low volatility strategy
├── optimization/         # Portfolio optimization algorithms
├── montecarlo/          # Monte Carlo simulations
├── byma/                # Argentine market data (BYMA)
├── cme/                 # CME futures and rates
└── output/              # Generated reports and results
```

## Output

Results are saved to `/output` including:
- Portfolio returns (CSV)
- Cumulative performance (CSV)
- Covariance matrices
- Performance metrics (CAGR, volatility, max drawdown)

---

**Note**: This is a research and educational repository. Past performance does not guarantee future results.
