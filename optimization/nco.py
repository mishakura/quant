import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as hc

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2016-01-01'
end = '2019-12-30'

# Tickers of assets
assets = [
    'CEMB', 'EMB', 'EMLC', 'GLD', 'SLV', 'USO', 'VEA', 'IWM','EWZ',
    'IEMG', 'YPF', 'EEM','BTC-USD','ACWI','EFA','EWJ','FXI','GDX','IBB','ETH-USD','IEUR','IJH','IVE','IVW','VIG','XLB','XLC','XLI','XLK','XLP','XLRE','XLV','XLY'
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
    'ARGT': 0.40
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

# Building the portfolio object
port = hc.HCPortfolio(returns=Y)

# Estimate optimal portfolio:
model='NCO' # Could be HRP, HERC or NCO
codependence = 'pearson' # Correlation matrix used to group assets in clusters
method_cov = 'hist' # Covariance estimation technique
obj = "MinRisk" # Posible values are "MinRisk", "Utility", "Sharpe" and "ERC"
rm = 'MV' # Risk measure used, this time will be variance
rf = 0 # Risk free rate
l = 2 # Risk aversion factor, only usefull with "Utility" objective
linkage = 'ward' # Linkage method used to build clusters
max_k = 10 # Max number of clusters used in two difference gap statistic
leaf_order = True # Consider optimal order of leafs in dendrogram

w = port.optimization(model=model,
                      codependence=codependence,
                      method_cov=method_cov,
                      obj=obj,
                      rm=rm,
                      rf=rf,
                      l=l,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order)


print("\nOptimal NCO Portfolio Weights:")
print(w)

