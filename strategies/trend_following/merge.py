import os
import pandas as pd

def merge_indicators_csvs(folder_path='indicators', output_file='main.csv'):
    all_data = []
    csv_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.csv')]
    total = len(csv_files)
    for i, file in enumerate(csv_files):
        ticker = file.split('_')[0]  # Extract ticker before '_indicators'
        df = pd.read_csv(os.path.join(folder_path, file))
        df['ticker'] = ticker  # Add ticker column
        all_data.append(df)
        print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
    
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        # Reorder columns to make 'ticker' first
        cols = ['ticker'] + [col for col in merged_df.columns if col != 'ticker']
        merged_df = merged_df[cols]
        # Detect date column (check common names)
        date_col = None
        for col in ['date', 'Date', 'timestamp', 'Timestamp']:
            if col in merged_df.columns:
                date_col = col
                break
        # Sort chronologically by date (oldest first), then alphabetically by 'ticker'
        if date_col:
            merged_df[date_col] = pd.to_datetime(merged_df[date_col])
            merged_df = merged_df.sort_values(by=[date_col, 'ticker'], ascending=[True, True])
        else:
            # Fallback: sort alphabetically by ticker
            merged_df = merged_df.sort_values(by='ticker', ascending=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved to {output_file}")
    else:
        print("No CSV files found in the folder.")

# Call the function
merge_indicators_csvs()