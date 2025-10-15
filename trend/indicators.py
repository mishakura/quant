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
        df = pd.read_csv(csv_path)
        df['Max200'] = df['High'].rolling(window=200).max()
        df['Min200'] = df['Low'].rolling(window=200).min()
        df['Max100'] = df['High'].rolling(window=100).max()
        df['Min100'] = df['Low'].rolling(window=100).min()
        df['ATR25'] = compute_atr(df, window=25)
        indicator_path = os.path.join(indicator_folder, f'{ticker}_indicators.csv')
        df.to_csv(indicator_path, index=False)