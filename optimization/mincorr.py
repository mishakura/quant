import datetime
import os
import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
from scipy.optimize import minimize

TICKERS = [
	"ARKK","SPXL","XLE","GLD","XLF","FXI","QQQ","IBIT","IWDA","IEMG",
	"IEUR","IJH","ETHA","ACWI","EWZ","EFA","EEM","EWJ","IBB","IVW",
	"IVE","SLV","IWM","URA","DIA","SPY","XLC","XLY","XLP","XLV",
	"XLI","XLB","XLRE","XLK","CIBR","XLU","USO","ITA","TQQQ","VXX",
	"SMH","VIG","VEA","GDX","PSQ","SH"
]

def download_prices(tickers, start="2010-01-01", end=None, auto_adjust=True):
	if end is None:
		end = datetime.datetime.today().strftime("%Y-%m-%d")
	print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
	data = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust)["Close"]
	# If single ticker, make it a DataFrame
	if isinstance(data, pd.Series):
		data = data.to_frame()
	# Sort columns to a stable order
	data = data.reindex(columns=sorted(data.columns))
	return data

def returns_from_prices(prices, period="daily"):
	if period == "daily":
		rets = prices.pct_change().dropna()
	elif period == "log":
		rets = np.log(prices).diff().dropna()
	else:
		rets = prices.pct_change().dropna()
	return rets

def min_correlation_weights(returns, model='HRP', rm=None, codependence='pearson', linkage='ward', max_k=10, leaf_order=True, rf=0.0):
	"""Compute portfolio weights using Riskfolio's hierarchical methods (HRP or HERC).

	This uses Riskfolio's HCPortfolio.optimization which builds portfolios based on
	hierarchical clustering of the correlation matrix — a practical way to reduce
	pairwise correlations in the final allocation.

	Parameters
	- returns: DataFrame of asset returns
	- model: 'HRP' or 'HERC' (hierarchical risk parity / hierarchical equal risk contribution)
	- rm: risk measure used by HERC/HRP (optional)
	- codependence: correlation type ('pearson' by default)
	- linkage: clustering linkage ('ward' recommended)
	- max_k, leaf_order: clustering params
	- rf: risk-free rate (passed through)
	"""
	# Build a hierarchical portfolio object from Riskfolio
	hc_port = rp.HCPortfolio(returns=returns)

	# Call Riskfolio's optimization for hierarchical models
	# The exact accepted parameters depend on Riskfolio version; we mirror usage in this repo.
	try:
		w = hc_port.optimization(
			model=model,
			codependence=codependence,
			linkage=linkage,
			max_k=max_k,
			leaf_order=leaf_order,
			rf=rf,
			rm=rm
		)
	except TypeError:
		# Fallback if rm isn't accepted or signature differs
		w = hc_port.optimization(
			model=model,
			codependence=codependence,
			linkage=linkage,
			max_k=max_k,
			leaf_order=leaf_order,
			rf=rf
		)

	# Ensure a Series and align to returns columns
	if isinstance(w, pd.DataFrame):
		w = w.squeeze()
	w = w.reindex(returns.columns).fillna(0)
	# Normalize
	if w.sum() != 0:
		w = w / w.sum()
	return w

def summary_stats(returns, weights, rf=0.0):
	port_rets = returns.dot(weights)
	ann_ret = (1 + port_rets).prod() ** (252 / len(port_rets)) - 1
	ann_vol = port_rets.std() * np.sqrt(252)
	sharpe = (port_rets.mean() * 252 - rf) / ann_vol if ann_vol != 0 else np.nan
	return {'Annualized Return': ann_ret, 'Annualized Volatility': ann_vol, 'Sharpe': sharpe}

def main():
	here = os.path.dirname(__file__)
	prices = download_prices(TICKERS, start="2010-01-01")
	# drop columns with all NaN
	prices = prices.dropna(axis=1, how='all')
	print(f"Downloaded prices shape: {prices.shape}")

	rets = returns_from_prices(prices, period='daily')
	rets = rets.dropna(axis=1, how='all')
	print(f"Computed returns shape: {rets.shape}")

	# Use riskfolio Portfolio object to show available estimators (optional)
	try:
		port = rp.Portfolio(returns=rets)
		# estimate sample moments via riskfolio helper (keeps dependency used)
		cov = port.cov_mu(method='hist')[0]
		corr = rets.corr()
	except Exception:
		# fallback if riskfolio API differs or not available
		cov = rets.cov()
		corr = rets.corr()

	print("Running Riskfolio hierarchical optimizations (HRP + HERC)...")
	w_hrp = min_correlation_weights(rets, model='HRP', codependence='pearson', linkage='ward', max_k=10, leaf_order=True, rf=0.0)
	w_herc = min_correlation_weights(rets, model='HERC', codependence='pearson', linkage='ward', max_k=10, leaf_order=True, rf=0.0)

	print("HRP weights (top 10):")
	print(w_hrp.sort_values(ascending=False).head(10))
	print("HERC weights (top 10):")
	print(w_herc.sort_values(ascending=False).head(10))

	# Combined mean-of-means
	mean_all = pd.DataFrame({'HRP': w_hrp, 'HERC': w_herc}).mean(axis=1)
	mean_all = mean_all / mean_all.sum()

	print("Mean of HRP+HERC (top 10):")
	print(mean_all.sort_values(ascending=False).head(10))

	stats_hrp = summary_stats(rets, w_hrp)
	stats_herc = summary_stats(rets, w_herc)
	stats_mean = summary_stats(rets, mean_all)

	print("HRP stats:")
	for k, v in stats_hrp.items():
		print(f"{k}: {v:.4f}")
	print("HERC stats:")
	for k, v in stats_herc.items():
		print(f"{k}: {v:.4f}")
	print("MeanAll stats:")
	for k, v in stats_mean.items():
		print(f"{k}: {v:.4f}")

	# Save weights and correlation matrix
	out_dir = os.path.join(here, "mincorr_output")
	os.makedirs(out_dir, exist_ok=True)
	w_hrp.to_csv(os.path.join(out_dir, "mincorr_hrp_weights.csv"))
	w_herc.to_csv(os.path.join(out_dir, "mincorr_herc_weights.csv"))
	mean_all.to_csv(os.path.join(out_dir, "mincorr_mean_weights.csv"))
	corr.to_csv(os.path.join(out_dir, "asset_correlations.csv"))
	prices.to_csv(os.path.join(out_dir, "prices.csv"))
	print(f"Saved weights and data to {out_dir}")

if __name__ == '__main__':
	main()
