import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# In case of having data update every day (data, signals and indicators)

data_folder = os.path.join(os.path.dirname(__file__), 'data')
indicator_folder = os.path.join(os.path.dirname(__file__), 'indicators')
os.makedirs(indicator_folder, exist_ok=True)
today = datetime.today().strftime('%Y-%m-%d')

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

def add_signals_to_indicators(indicator_path, new_rows_only=True):
    df = pd.read_csv(indicator_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.sort_values("Date").reset_index(drop=True)
    # Ensure signal columns exist
    for col in ['Signal', 'Reason', 'EntryPrice', 'StopPrice', 'ExitPrice', 'PnL_Percent']:
        if col not in df.columns:
            if col == 'Reason':
                df[col] = ""
            else:
                df[col] = np.nan
    # Find rows missing signals (Signal column NaN)
    missing_mask = df['Signal'].isna()
    if not missing_mask.any():
        print(f"{os.path.basename(indicator_path)}: All signals present.")
        df.to_csv(indicator_path, index=False)
        return
    # Only process missing rows
    in_trade = False
    entry_price = None
    entry_atr = None
    for i in range(len(df)):
        if not missing_mask.iloc[i]:
            continue
        if i == 0:
            df.at[i, "Signal"] = 0
            df.at[i, "Reason"] = "no_prev"
            continue
        today_close = float(df.at[i, "Close"])
        prev_max200 = df.at[i - 1, "Max200"]
        prev_min100 = df.at[i - 1, "Min100"]
        prev_atr = df.at[i - 1, "ATR25"]
        if pd.isna(prev_max200) or pd.isna(prev_min100) or pd.isna(prev_atr):
            if in_trade:
                pass
            else:
                df.at[i, "Signal"] = 0
                df.at[i, "Reason"] = "missing_prev"
                continue
        if not in_trade:
            if today_close > prev_max200:
                in_trade = True
                entry_price = today_close
                entry_atr = prev_atr
                stop_level = entry_price - 3.0 * float(entry_atr)
                df.at[i, "Signal"] = 1
                df.at[i, "Reason"] = "enter"
                df.at[i, "EntryPrice"] = entry_price
                df.at[i, "StopPrice"] = stop_level
            else:
                df.at[i, "Signal"] = 0
                df.at[i, "Reason"] = "no_long"
        else:
            stop_level = entry_price - 3.0 * float(entry_atr)
            if today_close < stop_level:
                in_trade = False
                df.at[i, "Signal"] = 0
                df.at[i, "Reason"] = "stop_loss"
                df.at[i, "EntryPrice"] = np.nan
                df.at[i, "StopPrice"] = np.nan
                df.at[i, "ExitPrice"] = today_close
                df.at[i, "PnL_Percent"] = ((today_close - entry_price) / entry_price) * 100
                entry_price = None
                entry_atr = None
            elif today_close < prev_min100:
                in_trade = False
                df.at[i, "Signal"] = 0
                df.at[i, "Reason"] = "exit_min100"
                df.at[i, "EntryPrice"] = np.nan
                df.at[i, "StopPrice"] = np.nan
                df.at[i, "ExitPrice"] = today_close
                df.at[i, "PnL_Percent"] = ((today_close - entry_price) / entry_price) * 100
                entry_price = None
                entry_atr = None
            else:
                df.at[i, "Signal"] = 1
                df.at[i, "Reason"] = "hold"
                df.at[i, "EntryPrice"] = entry_price
                df.at[i, "StopPrice"] = stop_level
    df.to_csv(indicator_path, index=False)
    print(f"{os.path.basename(indicator_path)}: Signals updated for {missing_mask.sum()} new rows.")

for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        ticker = filename.replace('.csv', '')
        csv_path = os.path.join(data_folder, filename)
        indicator_path = os.path.join(indicator_folder, f'{ticker}_indicators.csv')
        # --- Step 1: Update price data ---
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"{ticker}: CSV is empty, skipping.")
            continue
        last_date = df['Date'].max()
        if last_date >= today:
            print(f"{ticker}: Already up to date.")
        else:
            start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            new_data = yf.download(ticker, start=start_date, end=today)
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = [col[0] for col in new_data.columns]
            if {'Close', 'High', 'Low'}.issubset(new_data.columns) and not new_data.empty:
                new_data = new_data.reset_index()[['Date', 'High', 'Low', 'Close']]
                new_data = new_data[~new_data['Date'].isin(df['Date'])]
                if not new_data.empty:
                    df = pd.concat([df, new_data], ignore_index=True)
                    df.to_csv(csv_path, index=False)
                    print(f"{ticker}: Added {len(new_data)} new rows.")
                else:
                    print(f"{ticker}: No new data to add.")
            else:
                print(f"{ticker}: No new High/Low/Close data available.")
        # --- Step 2: Update indicators ---
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        if os.path.exists(indicator_path):
            indicators_df = pd.read_csv(indicator_path)
            indicators_df['Date'] = pd.to_datetime(indicators_df['Date'], format='mixed', errors='coerce')
            last_indicator_date = indicators_df['Date'].max()
            last_indicator_date = pd.to_datetime(last_indicator_date)
            new_data = df[df['Date'] > last_indicator_date]
            if not new_data.empty:
                window_max = max(200, 100, 25)
                history = df[df['Date'] <= last_indicator_date].tail(window_max - 1)
                update_df = pd.concat([history, new_data], ignore_index=True)
                update_df['Max200'] = update_df['High'].rolling(window=200).max()
                update_df['Min200'] = update_df['Low'].rolling(window=200).min()
                update_df['Max100'] = update_df['High'].rolling(window=100).max()
                update_df['Min100'] = update_df['Low'].rolling(window=100).min()
                update_df['ATR25'] = compute_atr(update_df, window=25)
                update_df = update_df[update_df['Date'] > last_indicator_date]
                indicators_df = pd.concat([indicators_df, update_df], ignore_index=True)
                indicators_df.to_csv(indicator_path, index=False)
                print(f"{ticker}: Indicators updated with {len(update_df)} new rows.")
            else:
                print(f"{ticker}: Indicators already up to date.")
        else:
            df['Max200'] = df['High'].rolling(window=200).max()
            df['Min200'] = df['Low'].rolling(window=200).min()
            df['Max100'] = df['High'].rolling(window=100).max()
            df['Min100'] = df['Low'].rolling(window=100).min()
            df['ATR25'] = compute_atr(df, window=25)
            df.to_csv(indicator_path, index=False)
            print(f"{ticker}: Indicators file created.")
        # --- Step 3: Add signals only for new rows ---
        add_signals_to_indicators(indicator_path, new_rows_only=True)
