import requests
import pandas as pd
import numpy as np
import time
from tickers import tickers
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

API_KEY = "0KFXJZ7FMBVLXY8V"
TICKERS = tickers

def fetch_eps_quarterly(symbol, api_key):
    print(f"Downloading data for {symbol}...")
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if 'quarterlyEarnings' not in data:
            print(f"Error: No quarterlyEarnings for {symbol}")
            return None
        eps_df = pd.DataFrame(data['quarterlyEarnings'])
        eps_df['fiscalDateEnding'] = pd.to_datetime(eps_df['fiscalDateEnding'])
        eps_df['reportedEPS'] = pd.to_numeric(eps_df['reportedEPS'], errors='coerce')
        eps_df = eps_df.sort_values('fiscalDateEnding')
        return eps_df[['fiscalDateEnding', 'reportedEPS']].reset_index(drop=True)
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

def compute_yoy_changes(eps_df):
    eps_df['YoY_Change'] = eps_df['reportedEPS'].pct_change(4)
    return eps_df

def compute_sue(eps_df):
    sue_list = []
    for idx in range(8, len(eps_df)):
        last_8_yoy = eps_df.loc[idx-8:idx-1, 'YoY_Change'].dropna()
        if len(last_8_yoy) == 8:
            stdev = last_8_yoy.std(ddof=0)
            most_recent_yoy = eps_df.loc[idx, 'YoY_Change']
            sue = most_recent_yoy / stdev if stdev != 0 else np.nan
        else:
            sue = np.nan
        sue_list.append(sue)
    eps_df['SUE'] = [np.nan]*8 + sue_list
    return eps_df

def main():
    results = []
    for ticker in TICKERS:
        eps_df = fetch_eps_quarterly(ticker, API_KEY)
        if eps_df is None or eps_df.empty:
            print(f"Skipping {ticker} due to missing data.")
            time.sleep(1)
            continue
        eps_df = compute_yoy_changes(eps_df)
        eps_df = compute_sue(eps_df)
        eps_df = eps_df.sort_values('fiscalDateEnding', ascending=False).reset_index(drop=True)
        most_recent = eps_df.loc[0]
        if pd.isna(most_recent['SUE']):
            print(f"No valid SUE for {ticker}.")
            time.sleep(1)
            continue
        results.append({
            'Ticker': ticker,
            'Date': most_recent['fiscalDateEnding'].strftime('%Y-%m-%d'),
            'SUE': most_recent['SUE']
        })
        time.sleep(1)  # 1 second sleep = 60 calls/minute, safe for 75/min limit

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('SUE', ascending=True).reset_index(drop=True)
        print("\nMost recent SUE per ticker (ascending order):")
        print(df)
        # Filter top 10% SUE
        threshold = df['SUE'].quantile(0.9)
        top_10_df = df[df['SUE'] >= threshold].reset_index(drop=True)
        top_10_df.to_csv("top_10_percent_SUE_per_ticker.csv", index=False)
        print("\nTop 10% SUE per ticker (saved to top_10_percent_SUE_per_ticker.csv):")
        print(top_10_df)
    else:
        print("No valid SUE data collected.")

if __name__ == "__main__":
    main()


