import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import os
import sys

# ===========================
# 1. CONFIGURACIÓN DE ACTIVOS
# ===========================

# Activos a descargar de Yahoo Finance
yf_tickers = ['GLD']

# Nombre de la columna en el Excel para el activo local (tu Excel tiene columnas "Fecha" e "Precio")
excel_ticker = 'Close'

# ===========================
# 2. DESCARGA DE DATOS
# ===========================

yf_data = yf.download(yf_tickers, start='2015-01-01', end='2025-12-31')['Close']

script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(script_dir, 'indices.xlsx')
excel_data = pd.read_excel(excel_path, index_col=0, parse_dates=True)

if excel_ticker not in excel_data.columns:
    print(f"ERROR: columna '{excel_ticker}' no encontrada en {excel_path}. Columnas disponibles: {list(excel_data.columns)}")
    sys.exit(1)

# Combinar datos: renombrar la serie de Excel al mismo nombre que queremos en el portfolio
excel_series = excel_data[excel_ticker].rename('INDICE_LOCAL')

prices = pd.concat([yf_data, excel_series], axis=1).dropna()

# ===========================
# 3. EXPECTED RETURNS (MANUAL)
# ===========================
# Edita estos valores dentro del script según tus expectativas.
# Deben corresponder exactamente a los nombres de prices.columns.
expected_returns = {
    'GLD': 0.03,
    'INDICE_LOCAL': 0.06
}

# Validar que existan expectativas para cada activo
missing_mu = [c for c in prices.columns if c not in expected_returns]
if missing_mu:
    print(f"ERROR: faltan expected returns para: {missing_mu}")
    sys.exit(1)

mu = np.array([expected_returns[ticker] for ticker in prices.columns])

# ===========================
# 4. CÁLCULO DE COVARIANZA
# ===========================
returns = prices.pct_change().dropna()
cov_matrix = returns.cov() * 252

# ===========================
# 5. OPTIMIZACIÓN
# ===========================
def portfolio_variance(weights, cov):
    return weights.T @ cov @ weights

def portfolio_return(weights, exp_ret):
    return weights.T @ exp_ret

n_assets = len(prices.columns)
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds = tuple((0, 1) for _ in range(n_assets))
w0 = np.array([1/n_assets] * n_assets)

res = minimize(portfolio_variance, w0, args=(cov_matrix.values,), method='SLSQP', bounds=bounds, constraints=constraints)
if not res.success:
    print(f"ERROR: optimización falló: {res.message}")
    sys.exit(1)

optimal_weights = res.x

# ===========================
# 6. OUTPUT (solo las ponderaciones)
# ===========================
# Imprime únicamente las ponderaciones (requisito)
print(dict(zip(prices.columns, optimal_weights)))