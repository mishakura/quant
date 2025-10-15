import os
import pandas as pd

data_folder = os.path.join(os.path.dirname(__file__), 'data')
assets_path = os.path.join(os.path.dirname(__file__), 'assets.xlsx')

# Load assets
assets_df = pd.read_excel(assets_path)
tickers = assets_df.iloc[:, 0].dropna().tolist()

valid_tickers = []

for ticker in tickers:
    csv_path = os.path.join(data_folder, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        print(f"{ticker}: CSV file missing, will be removed from assets.")
        continue
    df = pd.read_csv(csv_path)
    # Check for empty file or missing essential columns
    if df.empty or not {'Date', 'High', 'Low', 'Close'}.issubset(df.columns):
        print(f"{ticker}: No valid data, will be removed from assets.")
        # Optionally, remove the file
        try:
            os.remove(csv_path)
        except Exception as e:
            print(f"Error removing {csv_path}: {e}")
        continue
    # Check for obvious errors (e.g., NaN values in price columns)
    if df[['High', 'Low', 'Close']].isnull().any().any():
        print(f"{ticker}: NaN values found, cleaning rows.")
        df = df.dropna(subset=['High', 'Low', 'Close'])
        df.to_csv(csv_path, index=False)
    valid_tickers.append(ticker)

# Update assets.xlsx with only valid tickers
assets_df = pd.DataFrame(valid_tickers, columns=[assets_df.columns[0]])
assets_df.to_excel(assets_path, index=False)
print("Assets file cleaned and updated.")