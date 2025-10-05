import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as hc
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import os

# List of all risk measures supported by Riskfolio for hierarchical models
RISK_MEASURES = [
    'MV',      # Variance
    'MAD',     # Mean Absolute Deviation
    'MSV',     # Semi Variance
    'FLPM',    # First Lower Partial Moment
    'SLPM',    # Second Lower Partial Moment
    'CVaR',    # Conditional Value at Risk
    'VaR',     # Value at Risk
    'WR',      # Worst Realization
    'MDD',     # Maximum Drawdown
    'ADD',     # Average Drawdown
    'CDaR',    # Conditional Drawdown at Risk
    'EDaR',    # Entropic Drawdown at Risk
    'UCI'      # Ulcer Index
]

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



# Set risk-free rate before any stats calculations
rf = 0.03

# Calculando retornos para todos los activos
Y = data_aligned.pct_change().dropna()

# --- Calculate SPY stats and add as a portfolio ---
if 'SPY' in Y.columns:
    spy_rets = Y['SPY']
    spy_ann_return = (1 + spy_rets).prod() ** (252 / len(spy_rets)) - 1
    spy_ann_vol = spy_rets.std() * np.sqrt(252)
    spy_sharpe = (spy_rets.mean() * 252 - rf) / (spy_rets.std() * np.sqrt(252))
    spy_sortino = (spy_rets.mean() * 252 - rf) / (spy_rets[spy_rets < 0].std() * np.sqrt(252)) if len(spy_rets[spy_rets < 0]) > 0 else float('nan')
    spy_running_max = (spy_rets + 1).cumprod().cummax()
    spy_cumulative = (spy_rets + 1).cumprod()
    spy_drawdown = spy_cumulative / spy_running_max - 1
    spy_max_dd = spy_drawdown.min()
    spy_dd_periods = spy_drawdown < 0
    spy_dd_groups = (spy_dd_periods != spy_dd_periods.shift()).cumsum()
    spy_dd_lengths = spy_drawdown[spy_dd_periods].groupby(spy_dd_groups).size()
    spy_max_dd_duration = spy_dd_lengths.max() if not spy_dd_lengths.empty else 0
    spy_avg_dd = spy_drawdown[spy_dd_periods].groupby(spy_dd_groups).mean().mean() if not spy_dd_lengths.empty else 0
    spy_calmar = spy_ann_return / abs(spy_max_dd) if spy_max_dd != 0 else float('nan')
    spy_omega = (spy_rets[spy_rets > rf/252].sum() / abs(spy_rets[spy_rets <= rf/252].sum())) if (spy_rets[spy_rets <= rf/252].sum() != 0) else float('nan')
    spy_skewness = skew(spy_rets)
    spy_kurt = kurtosis(spy_rets)
    spy_tail_ratio = (spy_rets.quantile(0.95) / abs(spy_rets.quantile(0.05))) if spy_rets.quantile(0.05) != 0 else float('nan')
    spy_win_rate = (spy_rets > 0).mean()
    spy_total_return = (1 + spy_rets).prod() - 1
    spy_var_5 = spy_rets.quantile(0.05)
    spy_cvar_5 = spy_rets[spy_rets <= spy_var_5].mean()
    spy_downside_dev = spy_rets[spy_rets < 0].std() * (252 ** 0.5)
    spy_pct_below_2 = (spy_rets < -0.02).mean()
    # Beta and Jensen's alpha of SPY vs itself
    spy_beta = 1.0
    spy_jensen_alpha = 0.0
    spy_stat = {
        'Risk Measure': 'SPY',
        'Annualized Return': spy_ann_return,
        'Annualized Volatility': spy_ann_vol,
        'Sharpe Ratio': spy_sharpe,
        'Sortino Ratio': spy_sortino,
        'Max Drawdown': spy_max_dd,
        'Max Drawdown (abs)': abs(spy_max_dd),
        'Average Drawdown': spy_avg_dd,
        'Max Drawdown Duration': spy_max_dd_duration,
        'Calmar Ratio': spy_calmar,
        'Omega Ratio': spy_omega,
        'Skew': spy_skewness,
        'Kurtosis': spy_kurt,
        'Tail Ratio': spy_tail_ratio,
        'VaR 5%': spy_var_5,
        'CVaR 5%': spy_cvar_5,
        'Downside Deviation': spy_downside_dev,
        '% Returns < -2%': spy_pct_below_2,
        'Win Rate': spy_win_rate,
        'Total Return': spy_total_return,
        'Beta vs SPY': spy_beta,
        "Jensen's Alpha": spy_jensen_alpha
    }
    print("\nSPY stats:")
    for k, v in spy_stat.items():
        if k != 'Risk Measure':
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
else:
    spy_rets = None

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


# Try all risk measures and save results to Excel (one sheet per risk measure)

model = 'HRP'
correlation = 'pearson'
linkage = 'ward'
max_k = 10
leaf_order = True




results = {}
stats = []

# Add SPY as a portfolio for comparison
if 'SPY' in Y.columns:
    stats.append(spy_stat)

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
        print(f"\nOptimal HRP Portfolio Weights for {rm}:")
        print(w)
        # Compute portfolio returns
        weights = w.values.flatten()
        port_rets = Y.dot(weights)
        # Compute stats with numpy/pandas/scipy only
        ann_return = (1 + port_rets).prod() ** (252 / len(port_rets)) - 1
        ann_vol = port_rets.std() * (252 ** 0.5)
        sharpe = (port_rets.mean() * 252 - rf) / (port_rets.std() * (252 ** 0.5))
        downside = port_rets[port_rets < 0]
        sortino = (port_rets.mean() * 252 - rf) / (downside.std() * (252 ** 0.5)) if len(downside) > 0 else float('nan')
        running_max = (port_rets + 1).cumprod().cummax()
        cumulative = (port_rets + 1).cumprod()
        drawdown = cumulative / running_max - 1
        max_dd = drawdown.min()
        # Average drawdown
        dd_periods = drawdown < 0
        dd_groups = (dd_periods != dd_periods.shift()).cumsum()
        dd_lengths = drawdown[dd_periods].groupby(dd_groups).size()
        max_dd_duration = dd_lengths.max() if not dd_lengths.empty else 0
        avg_dd = drawdown[dd_periods].groupby(dd_groups).mean().mean() if not dd_lengths.empty else 0
        calmar = ann_return / abs(max_dd) if max_dd != 0 else float('nan')
        omega = (port_rets[port_rets > rf/252].sum() / abs(port_rets[port_rets <= rf/252].sum())) if (port_rets[port_rets <= rf/252].sum() != 0) else float('nan')
        skewness = skew(port_rets)
        kurt = kurtosis(port_rets)
        tail_ratio = (port_rets.quantile(0.95) / abs(port_rets.quantile(0.05))) if port_rets.quantile(0.05) != 0 else float('nan')
        win_rate = (port_rets > 0).mean()
        total_return = (1 + port_rets).prod() - 1
        # Tail risk metrics
        var_5 = port_rets.quantile(0.05)
        cvar_5 = port_rets[port_rets <= var_5].mean()
        downside_dev = port_rets[port_rets < 0].std() * (252 ** 0.5)
        pct_below_2 = (port_rets < -0.02).mean()

        # --- Beta and Jensen's Alpha vs SPY ---
        if spy_rets is not None:
            # Align lengths
            aligned = pd.concat([port_rets, spy_rets], axis=1, join='inner').dropna()
            port_aligned = aligned.iloc[:,0]
            spy_aligned = aligned.iloc[:,1]
            # Beta: cov(port, spy) / var(spy)
            beta = np.cov(port_aligned, spy_aligned)[0,1] / np.var(spy_aligned)
            # Jensen's alpha: (Rp - Rf) - beta * (Rspy - Rf)
            port_ann_excess = port_aligned.mean() * 252 - rf
            spy_ann_excess = spy_aligned.mean() * 252 - rf
            jensen_alpha = port_ann_excess - beta * spy_ann_excess
        else:
            beta = float('nan')
            jensen_alpha = float('nan')

        stat = {
            'Risk Measure': rm,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Max Drawdown (abs)': abs(max_dd),
            'Average Drawdown': avg_dd,
            'Max Drawdown Duration': max_dd_duration,
            'Calmar Ratio': calmar,
            'Omega Ratio': omega,
            'Skew': skewness,
            'Kurtosis': kurt,
            'Tail Ratio': tail_ratio,
            'VaR 5%': var_5,
            'CVaR 5%': cvar_5,
            'Downside Deviation': downside_dev,
            '% Returns < -2%': pct_below_2,
            'Win Rate': win_rate,
            'Total Return': total_return,
            'Beta vs SPY': beta,
            "Jensen's Alpha": jensen_alpha
        }
        stats.append(stat)
    except Exception as e:
        print(f"\nRisk measure {rm} failed: {e}")

# Save all weights and stats to Excel
with pd.ExcelWriter("hrp_portfolio_weights_all_rm.xlsx") as writer:
    for rm, w in results.items():
        w.to_excel(writer, sheet_name=rm)
    # Save stats as a new sheet
    stats_df = pd.DataFrame(stats)
    stats_df.to_excel(writer, sheet_name="Stats", index=False)
print("\nAll HRP weights and stats saved to hrp_portfolio_weights_all_rm.xlsx")

# Print the annualized mean return of each asset
print("\nAnnualized mean return for each asset:")
for asset in Y.columns:
    mean_ret = Y[asset].mean() * 252
    print(f"{asset}: {mean_ret:.2%}")

