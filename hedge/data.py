import yfinance as yf
import pandas as pd
import os

# List of tickers
tickers = ['SPY']

# Base directory and data folder
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# Download data for each ticker from 1930-01-01
data_dict = {}
for ticker in tickers:
    try:
        df = yf.download(ticker, start='1930-01-01')
        if not df.empty:
            df = df[['Open', 'High', 'Low', 'Close']].reset_index()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
            df['Date'] = pd.to_datetime(df['Date'])
            data_dict[ticker] = df
            print(f"Downloaded data for {ticker}: {len(df)} rows")
        else:
            print(f"No data for {ticker}")
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

# Find the common starting date: the latest start date among all tickers
start_dates = [df['Date'].min() for df in data_dict.values()]
common_start = max(start_dates)
print(f"Common starting date: {common_start}")

# Slice each df from the common start and save to CSV
for ticker, df in data_dict.items():
    df_filtered = df[df['Date'] >= common_start].copy()
    csv_path = os.path.join(data_dir, f'{ticker}.csv')
    df_filtered.to_csv(csv_path, index=False)
    print(f"Saved {ticker} data to {csv_path} ({len(df_filtered)} rows)")