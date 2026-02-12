import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as hc
import os
import matplotlib.pyplot as plt

from quant.utils.constants import DEFAULT_RISK_FREE_RATE, RISK_MEASURES, TRADING_DAYS_PER_YEAR
from quant.analytics.performance import performance_summary

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '1900-01-01'
end = '2025-12-30'

""" assets = [
    "ARKK", "SPXL", "XLE", "GLD", "XLF", "FXI", "QQQ", "BTC-USD",
    "IEMG", "IEUR", "IJH", "ETH-USD", "ACWI", "EWZ", "EFA",
    "EEM", "EWJ", "IBB", "IVW", "IVE", "SLV", "IWM", "URA",
    "DIA", "SPY", "XLC", "XLY", "XLP", "XLV", "XLI", "XLB",
    "XLRE", "XLK", "CIBR", "XLU", "USO", "ITA", "TQQQ", "VXX",
    "SMH", "VIG", "VEA", "GDX", "PSQ", "SH", "ARGT"
] """

assets = ["TLT","HYG","PSP","DBC","DBMF","REMX", "QVAL",""]
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
all_assets = assets



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
rf = DEFAULT_RISK_FREE_RATE

# Calculando retornos para todos los activos
Y = data_aligned.pct_change().dropna()


def _compute_stats(port_rets, rm_label, spy_rets=None, risk_free_rate=DEFAULT_RISK_FREE_RATE):
    """Compute stats dict for a return series using quant.analytics.performance.

    Parameters
    ----------
    port_rets : pd.Series
        Portfolio (or benchmark) simple daily returns.
    rm_label : str
        Label for the 'Risk Measure' key (e.g. risk measure code or 'SPY').
    spy_rets : pd.Series or None
        SPY benchmark returns for beta/alpha computation.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    dict
        Stats dictionary matching the existing output format.
    """
    summary = performance_summary(
        port_rets,
        benchmark_returns=spy_rets,
        risk_free_rate=risk_free_rate,
    )
    gmean_daily = (
        (1 + port_rets).prod() ** (1 / len(port_rets)) - 1
        if len(port_rets) > 0
        else float('nan')
    )
    stat = {
        'Risk Measure': rm_label,
        'Annualized Return': summary['Annualized Return'],
        'Geometric Mean (daily)': gmean_daily,
        'Geometric Mean (annualized)': summary['Annualized Return'],
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
        'Total Return': summary['Total Return'],
        'Beta vs SPY': summary['Beta'],
        "Jensen's Alpha": summary["Jensen's Alpha"],
    }
    return stat


# --- SPY benchmark handling ---
# Behavior:
#  - Always obtain SPY returns for benchmarking (spy_rets) when possible.
#  - Only add SPY into data_aligned and Y (i.e. include SPY as a portfolio asset)
#    when the user explicitly included "SPY" in the assets/all_assets list.
spy_rets = None
spy_stat = None

try:
    if 'SPY' in all_assets:
        # User requested SPY as an asset -> ensure it's present in data_aligned/Y
        if 'SPY' in Y.columns:
            spy_rets = Y['SPY']
            print("Using SPY provided in assets as portfolio asset and benchmark.")
        else:
            print("SPY listed in assets but not present in downloaded data — downloading SPY and adding to portfolio.")
            spy_prices = yf.download('SPY', start=data_aligned.index[0], end=data_aligned.index[-1], auto_adjust=True)["Close"]
            spy_prices = spy_prices.reindex(data_aligned.index).ffill()
            data_aligned['SPY'] = spy_prices
            # Recompute returns for portfolio (SPY now part of portfolio returns)
            Y = data_aligned.pct_change().dropna()
            if 'SPY' in Y.columns:
                spy_rets = Y['SPY']
                print("SPY downloaded and added to portfolio returns.")
            else:
                spy_rets = None
                print("SPY download succeeded but no overlapping returns after alignment; SPY not added.")
    else:
        # SPY not in user's asset list -> download only for benchmark, do NOT add to portfolio/data_aligned/Y
        print("SPY not in provided assets — downloading SPY only as benchmark (won't be added to portfolio).")
        spy_prices = yf.download('SPY', start=data_aligned.index[0], end=data_aligned.index[-1], auto_adjust=True)["Close"]
        spy_prices = spy_prices.reindex(data_aligned.index).ffill()
        temp_rets = spy_prices.pct_change().dropna()
        # align benchmark to portfolio returns index (Y)
        if not temp_rets.empty:
            spy_rets = temp_rets.reindex(Y.index).dropna()
            if spy_rets.empty:
                spy_rets = None
                print("Downloaded SPY has no overlapping returns with portfolio dates; skipping SPY benchmark.")
        else:
            spy_rets = None
            print("Downloaded SPY has no data for the requested range; skipping SPY benchmark.")
except Exception as e:
    spy_rets = None
    print(f"Failed to obtain SPY benchmark/asset: {e}")

# Compute SPY stats for benchmark display if available (do NOT modify portfolio/state here)
if spy_rets is not None:
    # Ensure spy_rets is a 1-D Series (squeeze DataFrame if needed) and drop NA
    if isinstance(spy_rets, pd.DataFrame):
        spy_rets = spy_rets.squeeze()
    spy_rets = pd.Series(spy_rets).dropna()

    spy_stat = _compute_stats(spy_rets, 'SPY', spy_rets=None, risk_free_rate=rf)
    # SPY vs itself: beta = 1, alpha = 0
    spy_stat['Beta vs SPY'] = 1.0
    spy_stat["Jensen's Alpha"] = 0.0

    print("\nSPY benchmark stats:")
    for k, v in spy_stat.items():
        if k != 'Risk Measure':
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
else:
    print("No SPY benchmark available; beta and Jensen's alpha will be NaN.")

# Output annualized standard deviation for all assets
print("\nAnnualized standard deviation for all assets:")
for asset in Y.columns:
    ann_std = Y[asset].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    print(f"{asset}: {ann_std:.2%}")

# Plot dendrogram using Riskfolio and save the figure
fig, ax = plt.subplots(figsize=(12, 8))
hc.plot_dendrogram(returns=Y,
                   codependence='pearson',
                   linkage='ward',
                   k=None,
                   max_k=10,
                   leaf_order=True,
                   ax=ax)
ax.set_title('Asset Dendrogram (pearson, single)')
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'dendrogram.png')
plt.savefig(output_path, dpi=300)
print(f"Dendrogram saved to {output_path}")
try:
    plt.show()
except Exception:
    # In non-interactive environments plt.show() may fail; ignore
    pass

# Building the portfolio object
port = hc.HCPortfolio(returns=Y)

# --- Add defaults for variables used in port.optimization to avoid NameError ---
# These match the dendrogram call used earlier; adjust if you prefer different settings.
correlation = 'pearson'
linkage = 'ward'
max_k = 10
leaf_order = True

# Replace the single-model results block with a two-model (HRP + HERC) workflow,
# per-model means, and a combined mean-of-means added to stats and saved.

# Models to run
MODELS = ['HRP','HERC']

# Collect raw weights per model -> per risk measure
results_by_model = {model: {} for model in MODELS}
for model in MODELS:
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
            # ensure we store a Series (squeeze single-column DataFrames)
            results_by_model[model][rm] = w.squeeze() if isinstance(w, pd.DataFrame) else w
            print(f"\nOptimal {model} Portfolio Weights for {rm}:")
            print(results_by_model[model][rm])
        except Exception as e:
            print(f"\n{model} risk measure {rm} failed: {e}")

# Compute per-model mean weights (mean across RMs) BEFORE stats
model_means = {}
for model, model_results in results_by_model.items():
    if not model_results:
        print(f"No results for model {model}, skipping mean computation.")
        continue
    weights_df = pd.DataFrame({rm: w for rm, w in model_results.items()})
    # align to asset order and compute mean
    weights_df = weights_df.reindex(Y.columns)
    mean_weights = weights_df.mean(axis=1)
    total = mean_weights.sum()
    mean_weights_normalized = mean_weights / total if total != 0 else mean_weights
    model_means[model] = mean_weights_normalized
    print(f"\nMean weights for {model}:")
    print(mean_weights_normalized)

# Compute combined mean across models (mean of the model_means)
mean_of_means = None
if model_means:
    mom_df = pd.DataFrame(model_means).reindex(Y.columns)
    mean_of_means = mom_df.mean(axis=1)
    total = mean_of_means.sum()
    mean_of_means_normalized = mean_of_means / total if total != 0 else mean_of_means
    print("\nMean of model means (HRP + HERC):")
    print(mean_of_means_normalized)
else:
    print("No model means available to compute mean-of-means.")

# Flatten results into a single dict for saving/stats, include per-model RMs and model means and the combined mean
results = {}
for model, model_results in results_by_model.items():
    for rm, w in model_results.items():
        sheet_name = f"{model}_{rm}"
        results[sheet_name] = w.reindex(Y.columns).fillna(0) if isinstance(w, pd.Series) else w

# Add per-model mean sheets
for model, mean_w in model_means.items():
    results[f"Mean{model}"] = mean_w.reindex(Y.columns).fillna(0)

# Add combined mean-of-means
if mean_of_means is not None:
    results["MeanAll"] = mean_of_means_normalized.reindex(Y.columns).fillna(0)

# Now compute stats for each result (including SPY and the new ensemble sheets)
stats = []
# Always include SPY benchmark stats if we were able to obtain them,
# even when SPY is not part of the portfolio asset list.
if spy_stat is not None:
    stats.append(spy_stat)

for rm, w in results.items():
    try:
        # Normalize/align weight vector to asset order
        if isinstance(w, pd.DataFrame):
            weights = w.squeeze()
        else:
            weights = w
        # if it's a Series, ensure reindex; if it's array-like convert to Series with columns order
        if isinstance(weights, (pd.Series, pd.Index)):
            weights = weights.reindex(Y.columns).fillna(0)
            weight_vals = weights.values
        else:
            # numeric array - assume same order as Y.columns
            weight_vals = np.asarray(weights).flatten()
            if weight_vals.size != len(Y.columns):
                # fallback: try to align by index if available
                try:
                    weights = pd.Series(weight_vals, index=Y.columns)
                    weight_vals = weights.values
                except Exception:
                    # cannot align; pad/truncate
                    arr = np.zeros(len(Y.columns))
                    arr[:min(len(arr), len(weight_vals))] = weight_vals[:len(arr)]
                    weight_vals = arr

        port_rets = Y.dot(weight_vals)

        stat = _compute_stats(port_rets, rm, spy_rets=spy_rets, risk_free_rate=rf)
        stats.append(stat)
    except Exception as e:
        print(f"\nStats for {rm} failed: {e}")

# --- SPY: do NOT download/add SPY here (benchmark handled later) ---
# Previously we downloaded and inserted SPY into data_aligned which caused SPY to
# appear in the portfolio even when not included in the user's assets list.
# Keep spy_rets/spy_stat variables defined for later benchmark usage.
spy_rets = None
spy_stat = None

try:
    if 'SPY' in Y.columns:
        # User already included SPY as an asset; use that column as benchmark (but keep it as asset)
        spy_rets = Y['SPY']
        print("Using SPY from provided assets as benchmark.")
    else:
        # Download SPY prices for same date range as our aligned data, but do NOT add to data_aligned/Y
        print("SPY not in provided assets — downloading SPY only as benchmark (won't be added to portfolio).")
        spy_prices = yf.download('SPY', start=data_aligned.index[0], end=data_aligned.index[-1], auto_adjust=True)["Close"]
        spy_prices = spy_prices.reindex(data_aligned.index).ffill()
        spy_rets = spy_prices.pct_change().dropna()
        if spy_rets.empty:
            spy_rets = None
            print("Downloaded SPY has no overlapping returns with portfolio dates; skipping SPY benchmark.")
except Exception as e:
    spy_rets = None
    print(f"Failed to obtain SPY benchmark: {e}")

if spy_rets is not None:
    # Ensure series (squeeze DataFrame -> Series if needed)
    if isinstance(spy_rets, pd.DataFrame):
        spy_rets = spy_rets.squeeze()
    spy_rets = pd.Series(spy_rets).dropna()

    spy_stat = _compute_stats(spy_rets, 'SPY', spy_rets=None, risk_free_rate=rf)
    # SPY vs itself: beta = 1, alpha = 0
    spy_stat['Beta vs SPY'] = 1.0
    spy_stat["Jensen's Alpha"] = 0.0

    print("\nSPY benchmark stats:")
    for k, v in spy_stat.items():
        if k != 'Risk Measure':
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
else:
    print("No SPY benchmark available; beta and Jensen's alpha will be NaN.")

# Save all weights and stats to Excel (each key in results becomes a sheet)
with pd.ExcelWriter("hrp_portfolio_weights_all_rm.xlsx") as writer:
    for rm, w in results.items():
        # if Series/DataFrame write directly, else convert to Series
        if isinstance(w, pd.Series):
            w.to_excel(writer, sheet_name=str(rm))
        elif isinstance(w, pd.DataFrame):
            w.to_excel(writer, sheet_name=str(rm))
        else:
            try:
                pd.Series(w, index=Y.columns).to_excel(writer, sheet_name=str(rm))
            except Exception:
                # fallback: write empty sheet
                pd.DataFrame().to_excel(writer, sheet_name=str(rm))

    # Save stats as a new sheet
    stats_df = pd.DataFrame(stats)
    stats_df.to_excel(writer, sheet_name="Stats", index=False)

    # Save per-model mean weights and combined mean separately for convenience
    for model, mean_w in model_means.items():
        pd.DataFrame(mean_w, columns=['Mean Weight']).to_excel(writer, sheet_name=f"Mean{model}Weights")

    if mean_of_means is not None:
        pd.DataFrame(mean_of_means_normalized, columns=['Mean Weight']).to_excel(writer, sheet_name="MeanAllWeights")

print("\nAll HRP + HERC weights and stats saved to hrp_portfolio_weights_all_rm.xlsx")

# Print the annualized mean return of each asset
print("\nAnnualized mean return for each asset:")
for asset in Y.columns:
    mean_ret = Y[asset].mean() * TRADING_DAYS_PER_YEAR
    print(f"{asset}: {mean_ret:.2%}")
