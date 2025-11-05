import os
import pandas as pd

data_folder = os.path.join(os.path.dirname(__file__), 'data')
assets_path = os.path.join(os.path.dirname(__file__), 'assets.xlsx')

# Load assets (do NOT modify assets.xlsx)
assets_df = pd.read_excel(assets_path)
tickers = assets_df.iloc[:, 0].dropna().tolist()

invalid_assets = []

for ticker in tickers:
    csv_path = os.path.join(data_folder, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        print(f"{ticker}: CSV file missing (not downloaded).")
        invalid_assets.append(ticker)
        continue
    df = pd.read_csv(csv_path)
    # Check for empty file or missing essential columns
    if df.empty or not {'Date', 'High', 'Low', 'Close'}.issubset(df.columns):
        print(f"{ticker}: No valid data -> removing CSV.")
        try:
            os.remove(csv_path)
        except Exception as e:
            print(f"Error removing {csv_path}: {e}")
        invalid_assets.append(ticker)
        continue
    # Check for obvious errors (e.g., NaN values in price columns)
    if df[['High', 'Low', 'Close']].isnull().any().any():
        print(f"{ticker}: NaN values found -> removing CSV.")
        try:
            os.remove(csv_path)
        except Exception as e:
            print(f"Error removing {csv_path}: {e}")
        invalid_assets.append(ticker)
        continue
    # If reached here, CSV is considered valid
    print(f"{ticker}: OK")

# Report summary of not-clean assets (assets.xlsx left unchanged)
if invalid_assets:
    print("\nAssets with missing/invalid data (CSV removed if present):")
    for a in invalid_assets:
        print(f" - {a}")
else:
    print("All assets appear clean.")