import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as hc

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2000-01-01'
end = '2025-12-30'

# Tickers of assets
assets = [
    'CEMB', 'EMB', 'EMLC', 'GLD', 'SLV', 'USO', 'VEA', 'IWM','EWZ',
    'IEMG', 'YPF','BTC-USD','FXI','GDX','IBB','ETH-USD','IJH','XLB','XLC','XLI','XLK','XLP','XLRE','XLV','XLY'
]
assets.sort()

# Downloading data
data = yf.download(assets, start=start, end=end)
data = data.loc[:, ('Close', slice(None))]
data.columns = assets

# Calculating returns
Y = data[assets].pct_change().dropna()

# Target annualized volatilities
target_vols = {
    'EMLC': 0.56,
    'EMB': 0.21,
    'CEMB': 0.07,
}

# Calculate current daily std
current_stds = Y.std()

# Scale returns for specified assets
for asset, target_annual_std in target_vols.items():
    if asset in Y.columns:
        target_daily_std = target_annual_std / np.sqrt(252)
        scaling_factor = target_daily_std / current_stds[asset]
        Y[asset] = Y[asset] * scaling_factor

print("Scaled returns (first 5 rows):")
print(Y.head())

# Optional: Check new annualized stds
print("\nNew annualized stds for scaled assets:")
for asset in target_vols:
    if asset in Y.columns:
        print(f"{asset}: {Y[asset].std() * np.sqrt(252):.2%}")

# Output annualized standard deviation for all assets
print("\nAnnualized standard deviation for all assets:")
for asset in Y.columns:
    ann_std = Y[asset].std() * np.sqrt(252)
    print(f"{asset}: {ann_std:.2%}")

ax = hc.plot_dendrogram(returns=Y,
                         codependence='pearson',
                         linkage='single',
                         k=None,
                         max_k=10,
                         leaf_order=True,
                         ax=None)

# Building the portfolio object
port = hc.HCPortfolio(returns=Y)

# Estimate optimal portfolio:

model='HRP' # Could be HRP or HERC
correlation = 'pearson' # Correlation matrix used to group assets in clusters
rm = 'MV' # Risk measure used, this time will be variance
rf = 0 # Risk free rate
linkage = 'ward' # Linkage method used to build clusters
max_k = 10 # Max number of clusters used in two difference gap statistic
leaf_order = True # Consider optimal order of leafs in dendrogram

w = port.optimization(model=model,
                      codependence=correlation,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order)

print("\nOptimal HERC Portfolio Weights:")
print(w)

# Output weights to Excel
w.to_excel("herc_portfolio_weights.xlsx")
print("\nWeights saved to herc_weights.xlsx")

