import yfinance as yf
import os
import pandas as pd

# Define the ticker symbol
ticker = 'VXX'

# Download historical data from 1930 onwards (note: VXX data starts in 2009)
data = yf.download(ticker, start='1930-01-01')

# Select Date and Close price
data = data[['Close']].reset_index()
data.columns = ['Date', 'Price']

# Ensure the 'data' folder exists
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Save to CSV
csv_path = os.path.join(data_dir, 'vxx_data.csv')
data.to_csv(csv_path, index=False)

print(f"Data saved to {csv_path}")