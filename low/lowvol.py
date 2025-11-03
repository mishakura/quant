import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tickers import tickers

# Define date range
end_date = "2025-11-01"
start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=3*365)).strftime("%Y-%m-%d")

# Download adjusted close prices for each ticker one by one
all_data = {}
N = len(tickers)
for idx, ticker in enumerate(tickers, 1):
    print(f"Downloading {ticker} ({idx}/{N})...")
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        if df is not None and not df.empty and "Adj Close" in df.columns:
            # Flatten the array if it's 2D to handle cases like PAM
            values = df["Adj Close"].values
            if len(values.shape) > 1:
                values = values.flatten()
            all_data[ticker] = pd.Series(values, index=df.index)
        else:
            print(f"No data for {ticker}, skipping.")
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

print("all_data type:", type(all_data))
print("all_data sample:", {k: type(v) for k, v in list(all_data.items())[:3]})
print("all_data length:", len(all_data))

if not all_data:
    raise ValueError("No data was downloaded for any ticker.")

# Fix: Use pd.concat to properly merge the data
data = pd.concat(all_data.values(), axis=1, keys=all_data.keys())

# Calculate daily returns
returns = data.pct_change(fill_method=None).dropna()

# Calculate standard deviation of returns for each ticker
volatility = returns.std()

# To annualize, add this line:
volatility = volatility * (252 ** 0.5)  # Annualize using sqrt(252)

# Select top 10 percentile of least volatile stocks
threshold = volatility.quantile(0.1)
low_vol_stocks = volatility[volatility <= threshold].sort_values()

# Output result to CSV
low_vol_stocks.to_csv("low_volatility_stocks.csv", header=["StdDev"])

print("Low volatility stocks saved to low_volatility_stocks.csv")