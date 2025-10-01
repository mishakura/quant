import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as hc
import os

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2000-01-01'
end = '2025-12-30'

# Tickers of assets
assets = [
    "VEA","IJH","IWM","IEMG","ARGT","BTC-USD","GDX","GLD","SLV","USO","URA"
]
assets.sort()

# Descargar precios de Yahoo Finance
data = yf.download(assets, start=start, end=end)["Close"]

# Leer precios desde el Excel 'indices.xlsx' y agregar los activos
excel_path = os.path.join(os.path.dirname(__file__), 'indices.xlsx')
extra_assets = []
if os.path.exists(excel_path):
    xl = pd.ExcelFile(excel_path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df = df.rename(columns={'Fecha': 'Date', 'Precio': sheet})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        # Unir con los datos descargados (por fecha)
        data = data.join(df[[sheet]], how='outer')
        extra_assets.append(sheet)
else:
    print("No se encontró el archivo indices.xlsx")

# Combinar todos los activos (tickers + hojas del Excel)
all_assets = assets + extra_assets

# Calculando retornos para todos los activos
Y = data[all_assets].pct_change().dropna()

# Calculate current daily std
current_stds = Y.std()

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
rf = 0.3 # Risk free rate
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

# Calculate expected portfolio return and risk
weights = w.values.flatten()
mean_returns = Y.mean() * 252  # Annualized mean returns
cov_matrix = Y.cov() * 252     # Annualized covariance matrix

# Expected portfolio return
portfolio_return = np.dot(weights, mean_returns)

# Expected portfolio risk (standard deviation)
portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

print(f"\nExpected annualized portfolio return: {portfolio_return:.2%}")
print(f"Expected annualized portfolio risk (std): {portfolio_risk:.2%}")

# Output weights to Excel
w.to_excel("herc_portfolio_weights.xlsx")
print("\nWeights saved to herc_portfolio_weights.xlsx")

print("\nEarliest available date for each asset:")
for asset in all_assets:
    print(f"{asset}: {data[asset].first_valid_index()}")

# Print the annualized mean return of each asset
print("\nAnnualized mean return for each asset:")
for asset in Y.columns:
    mean_ret = Y[asset].mean() * 252
    print(f"{asset}: {mean_ret:.2%}")

# Forzar desviación estándar anualizada de 8% para Galileo Event Driven
target_std = 0.08
asset_name = "Galileo Event Driven"
if asset_name in data.columns:
    # Calcular retornos diarios logarítmicos
    log_ret = np.log(data[asset_name] / data[asset_name].shift(1)).dropna()
    current_std = log_ret.std() * np.sqrt(252)
    if current_std > 0:
        scale = target_std / current_std
        # Escalar los retornos logarítmicos
        adj_log_ret = log_ret * scale
        # Reconstruir la serie de precios ajustada
        adj_prices = data[asset_name].dropna().iloc[0] * np.exp(adj_log_ret.cumsum())
        data.loc[adj_prices.index, asset_name] = adj_prices
        print(f"Desviación estándar anualizada de '{asset_name}' ajustada a 8%.")
    else:
        print(f"No se pudo ajustar la desviación estándar de '{asset_name}' (std actual = 0).")
else:
    print(f"'{asset_name}' no está en los datos.")

