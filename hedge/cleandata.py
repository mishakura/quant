import pandas as pd
import os

# Base directory and data folder
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')

# Get list of CSV files in data folder
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

for csv_file in csv_files:
    csv_path = os.path.join(data_dir, csv_file)
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by Date
        df.sort_values('Date', inplace=True)
        
        # Drop duplicate dates, keeping the first
        df.drop_duplicates(subset='Date', inplace=True)
        
        # Drop rows with NaN in Price
        df.dropna(subset=['Price'], inplace=True)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        # Save back to CSV
        df.to_csv(csv_path, index=False)
        
        print(f"Cleaned {csv_file}: {len(df)} rows")
    except Exception as e:
        print(f"Error cleaning {csv_file}: {e}")