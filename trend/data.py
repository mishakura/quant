import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# Read asset tickers from Excel file
assets_df = pd.read_excel('assets.xlsx')
tickers = assets_df.iloc[:, 0].dropna().tolist()  # Assumes tickers are in the first column

# Ensure data folder exists (inside 'trend')
data_folder = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_folder, exist_ok=True)

# Download data for each ticker
start_date = '1930-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    if {'Close', 'High', 'Low'}.issubset(df.columns):
        df = df.reset_index()[['Date', 'High', 'Low', 'Close']]
        csv_path = os.path.join(data_folder, f'{ticker}.csv')
        df.to_csv(csv_path, index=False, header=True)
        print(f'Saved {ticker} data to {csv_path}')
    else:
        print(f'No High/Low/Close data for {ticker}')
