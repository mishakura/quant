import os
import pandas as pd
import numpy as np

data_folder = os.path.join(os.path.dirname(__file__), 'data')
indicator_folder = os.path.join(os.path.dirname(__file__), 'indicators')
os.makedirs(indicator_folder, exist_ok=True)

def compute_atr(df, window=25):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        ticker = filename.replace('.csv', '')
        csv_path = os.path.join(data_folder, filename)
        indicator_path = os.path.join(indicator_folder, f'{ticker}_indicators.csv')

        df = pd.read_csv(csv_path)
        if os.path.exists(indicator_path):
            indicators_df = pd.read_csv(indicator_path)
            last_indicator_date = indicators_df['Date'].max()
            new_data = df[df['Date'] > last_indicator_date]
            if not new_data.empty:
                # Recompute indicators for the new data, but need enough history for rolling windows
                # Concatenate last (window-1) rows from old data with new data for correct rolling
                window_max = max(200, 100, 25)
                history = df[df['Date'] <= last_indicator_date].tail(window_max - 1)
                update_df = pd.concat([history, new_data], ignore_index=True)
                update_df['Max200'] = update_df['High'].rolling(window=200).max()
                update_df['Min200'] = update_df['Low'].rolling(window=200).min()
                update_df['Max100'] = update_df['High'].rolling(window=100).max()
                update_df['Min100'] = update_df['Low'].rolling(window=100).min()
                update_df['ATR25'] = compute_atr(update_df, window=25)
                # Only keep rows with new dates
                update_df = update_df[update_df['Date'] > last_indicator_date]
                # Append to indicators file
                indicators_df = pd.concat([indicators_df, update_df], ignore_index=True)
                indicators_df.to_csv(indicator_path, index=False)
                print(f"{ticker}: Indicators updated with {len(update_df)} new rows.")
            else:
                print(f"{ticker}: Indicators already up to date.")
        else:
            # No indicators file, compute for all data
            df['Max200'] = df['High'].rolling(window=200).max()
            df['Min200'] = df['Low'].rolling(window=200).min()
            df['Max100'] = df['High'].rolling(window=100).max()
            df['Min100'] = df['Low'].rolling(window=100).min()
            df['ATR25'] = compute_atr(df, window=25)
            df.to_csv(indicator_path, index=False)
            print(f"{ticker}: Indicators file created.")