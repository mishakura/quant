import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as hc
import os

from quant.utils.constants import DEFAULT_RISK_FREE_RATE, RISK_MEASURES, TRADING_DAYS_PER_YEAR
from quant.analytics.performance import performance_summary

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2000-01-01'
end = '2025-12-30'

assets = [
 "VEA","IJH","IWM","IEMG","YPF","BTC-USD","GDX","GLD","SLV","USO","URA","SPY"
]

assets.sort()

# Descargar precios de Yahoo Finance (auto_adjust=True para ajustar precios por splits/dividendos)
data = yf.download(assets, start=start, end=end, auto_adjust=True)["Close"]

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

print("\nEarliest available date for each asset:")
for asset in all_assets:
    first_date = data[asset].first_valid_index()
    print(f"{asset}: {first_date}")

# Find the latest start date (when ALL assets have data)
latest_start = data[all_assets].apply(lambda x: x.first_valid_index()).max()
print(f"\nLatest common start date: {latest_start}")

# Trim data to start from latest_start (so all assets have complete data)
data_aligned = data.loc[latest_start:, all_assets].copy()

# Drop any remaining rows with NaN (shouldn't be any after alignment)
data_aligned = data_aligned.dropna()

print(f"Using data from {data_aligned.index[0]} to {data_aligned.index[-1]}")
print(f"Total days: {len(data_aligned)}")

# Calculando retornos para todos los activos
Y = data_aligned.pct_change().dropna()

# Output annualized standard deviation for all assets
print("\nAnnualized standard deviation for all assets:")
for asset in Y.columns:
    ann_std = Y[asset].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
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


# Try all risk measures and save results to Excel (one sheet per risk measure)
model = 'NCO'
correlation = 'pearson'
rf = DEFAULT_RISK_FREE_RATE
linkage = 'ward'
max_k = 10
leaf_order = True



results = {}
stats = []

for rm in RISK_MEASURES:
    try:
        w = port.optimization(
            model=model,
            codependence=correlation,
            rm=rm,
            rf=rf,
            linkage=linkage,
            max_k=max_k,
            leaf_order=leaf_order
        )
        results[rm] = w
        print(f"\nOptimal nco Portfolio Weights for {rm}:")
        print(w)
        # Compute portfolio returns
        weights = w.values.flatten()
        port_rets = Y.dot(weights)
        # Compute stats using quant.analytics.performance
        summary = performance_summary(port_rets, risk_free_rate=rf)
        stat = {
            'Risk Measure': rm,
            'Annualized Return': summary['Annualized Return'],
            'Annualized Volatility': summary['Annualized Volatility'],
            'Sharpe Ratio': summary['Sharpe Ratio'],
            'Sortino Ratio': summary['Sortino Ratio'],
            'Max Drawdown': summary['Max Drawdown'],
            'Max Drawdown (abs)': abs(summary['Max Drawdown']),
            'Average Drawdown': summary['Average Drawdown'],
            'Max Drawdown Duration': summary['Max Drawdown Duration'],
            'Calmar Ratio': summary['Calmar Ratio'],
            'Omega Ratio': summary['Omega Ratio'],
            'Skew': summary['Skewness'],
            'Kurtosis': summary['Kurtosis'],
            'Tail Ratio': summary['Tail Ratio'],
            'VaR 5%': summary['VaR 95%'],
            'CVaR 5%': summary['CVaR 95%'],
            'Downside Deviation': summary['Downside Deviation'],
            '% Returns < -2%': (port_rets < -0.02).mean(),
            'Win Rate': summary['Win Rate'],
            'Total Return': summary['Total Return']
        }
        stats.append(stat)
    except Exception as e:
        print(f"\nRisk measure {rm} failed: {e}")

# Save all weights and stats to Excel
with pd.ExcelWriter("nco_portfolio_weights_all_rm.xlsx") as writer:
    for rm, w in results.items():
        w.to_excel(writer, sheet_name=rm)
    # Save stats as a new sheet
    stats_df = pd.DataFrame(stats)
    stats_df.to_excel(writer, sheet_name="Stats", index=False)
print("\nAll nco weights and stats saved to nco_portfolio_weights_all_rm.xlsx")

# Print the annualized mean return of each asset
print("\nAnnualized mean return for each asset:")
for asset in Y.columns:
    mean_ret = Y[asset].mean() * TRADING_DAYS_PER_YEAR
    print(f"{asset}: {mean_ret:.2%}")
