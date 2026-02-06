"""Portfolio backtest (long-only) using yfinance and local 'indices.xlsx' for ON.

- Downloads adjusted close prices from yfinance for most tickers.
- Reads 'indices.xlsx' sheet 'ON' with columns 'Date' and 'Precio' for ON.
- Computes daily returns, portfolio returns, covariance-based portfolio volatility
  (annualized) accounting for correlations, and basic performance metrics.

Usage: python portfolio.py

Dependencies: pandas, numpy, yfinance, openpyxl
"""

import math
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf


# Portfolio will be 3 assets:
# - AC equity: VT (50%)
# - AC bonds: BNDW (20%)
# - Alternatives: synthetic equal-weight index from several ETFs (30%)
ASSETS = ["VT", "BNDW", "ALT"]

# Alternative ETFs to build the equal-weight index from
ALT_TICKERS = ["VNQ", "REM", "IGF", "XTN", "WOOD", "DBC", "PSP", "QAI"]

# Percent weights (will be treated as percentages and converted to fractions)
WEIGHTS_PCT = {
	"VT": 50.0,
	"BNDW": 20.0,
	"ALT": 30.0,
}

TRADING_DAYS = 252


def load_prices(start=None, end=None, excel_path="indices.xlsx"):
	# We'll download VT and BNDW plus the alternative ETFs used to build the ALT index.
	download_tickers = ["VT", "BNDW"] + ALT_TICKERS
	prices = pd.DataFrame()
	if len(download_tickers) > 0:
		data = yf.download(download_tickers, start=start, end=end, progress=False, auto_adjust=True)
		if isinstance(data.columns, pd.MultiIndex):
			if "Close" in data:
				prices = data["Close"].copy()
			else:
				prices = data["Adj Close"] if "Adj Close" in data else data.iloc[:, data.columns.get_level_values(1) == "Close"]
		else:
			prices = data["Close"] if "Close" in data else data

	if prices.empty:
		raise RuntimeError("No price data downloaded from yfinance")

	prices.index = pd.to_datetime(prices.index)

	# Build equal-weight alternatives index from available ALT_TICKERS
	available_alts = [t for t in ALT_TICKERS if t in prices.columns]
	if len(available_alts) == 0:
		raise RuntimeError("None of the alternative tickers were found in downloaded data")

	# Compute daily returns for alt tickers and average (equal weight)
	alt_returns = prices[available_alts].pct_change()
	alt_mean_ret = alt_returns.mean(axis=1)
	alt_mean_ret = alt_mean_ret.dropna()

	# Build an index series starting at 1.0 (relative price) from cumulative returns
	alt_index = (1.0 + alt_mean_ret).cumprod()
	alt_index.name = "ALT"

	# Merge the ALT index into prices (keep only VT, BNDW, ALT)
	df = prices.join(alt_index, how="inner")

	# Ensure VT and BNDW exist
	missing = [t for t in ["VT", "BNDW"] if t not in df.columns]
	if missing:
		raise RuntimeError(f"Missing required tickers: {missing}")

	df = df[["VT", "BNDW", "ALT"]].sort_index()
	return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
	returns = prices.pct_change().dropna(how="all")
	returns = returns.dropna()
	return returns


def portfolio_metrics(returns: pd.DataFrame, weights_pct: dict):
	w = pd.Series(weights_pct).astype(float) / 100.0
	# align weights with returns columns
	w = w.reindex(returns.columns).fillna(0.0)

	# Portfolio daily returns from weighted sum
	portf_ret = returns.dot(w)

	# Covariance-based portfolio volatility
	cov = returns.cov()
	port_var_daily = float(w.values.T @ cov.values @ w.values)
	port_vol_daily = math.sqrt(port_var_daily)
	port_vol_annual = port_vol_daily * math.sqrt(TRADING_DAYS)

	# Validate by sample std of portfolio returns
	sample_vol_annual = portf_ret.std(ddof=1) * math.sqrt(TRADING_DAYS)

	# Performance
	cumulative = (1 + portf_ret).cumprod()
	total_return = cumulative.iloc[-1] - 1.0
	years = len(portf_ret) / TRADING_DAYS
	cagr = (cumulative.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
	running_max = cumulative.cummax()
	drawdown = cumulative / running_max - 1.0
	max_dd = drawdown.min()

	results = {
		"portfolio_returns": portf_ret,
		"covariance": cov,
		"port_vol_annual_cov": port_vol_annual,
		"port_vol_annual_sample": sample_vol_annual,
		"cumulative": cumulative,
		"total_return": total_return,
		"cagr": cagr,
		"max_drawdown": max_dd,
	}
	return results


def save_outputs(results: dict, out_prefix="portfolio_backtest"):
	Path("output").mkdir(exist_ok=True)
	results["portfolio_returns"].to_csv(Path("output") / f"{out_prefix}_returns.csv", header=["portfolio_return"]) 
	results["cumulative"].to_csv(Path("output") / f"{out_prefix}_cumulative.csv", header=["cumulative"]) 
	results["covariance"].to_csv(Path("output") / f"{out_prefix}_covariance.csv")


def main():
	# Load prices (auto range)
	prices = load_prices(excel_path=Path(__file__).parent / "indices.xlsx")
	if prices.empty:
		raise RuntimeError("No price data loaded")

	returns = compute_returns(prices)

	results = portfolio_metrics(returns, WEIGHTS_PCT)

	print("Annualized volatility (covariance):", round(results["port_vol_annual_cov"] * 100, 2), "%")
	print("Annualized volatility (sample):", round(results["port_vol_annual_sample"] * 100, 2), "%")
	print("CAGR:", f"{results['cagr']:.2%}")
	print("Total return:", f"{results['total_return']:.2%}")
	print("Max drawdown:", f"{results['max_drawdown']:.2%}")

	save_outputs(results)
	print("Saved outputs to ./output/")


if __name__ == "__main__":
	main()

