import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

data_folder = os.path.join(os.path.dirname(__file__), 'data')
today = datetime.today().strftime('%Y-%m-%d')

for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        ticker = filename.replace('.csv', '')
        csv_path = os.path.join(data_folder, filename)
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"{ticker}: CSV is empty, skipping.")
            continue
        last_date = df['Date'].max()
        if last_date >= today:
            print(f"{ticker}: Already up to date.")
            continue
        # Download new data from the day after last_date to today
        start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        new_data = yf.download(ticker, start=start_date, end=today)
        # Flatten multi-level columns if present
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [col[0] for col in new_data.columns]
        if {'Close', 'High', 'Low'}.issubset(new_data.columns) and not new_data.empty:
            new_data = new_data.reset_index()[['Date', 'High', 'Low', 'Close']]
            # Only add rows with dates not already in df
            new_data = new_data[~new_data['Date'].isin(df['Date'])]
            if not new_data.empty:
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"{ticker}: Added {len(new_data)} new rows.")
            else:
                print(f"{ticker}: No new data to add.")
        else:
            print(f"{ticker}: No new High/Low/Close data available.")