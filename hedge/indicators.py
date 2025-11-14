import pandas as pd
import os
import numpy as np

# Paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
indicators_dir = os.path.join(os.path.dirname(__file__), 'indicators')
os.makedirs(indicators_dir, exist_ok=True)

# Load VIX data
vix_file = os.path.join(data_dir, 'VIX.csv')
df = pd.read_csv(vix_file)

# Rename columns to match desired format
df = df.rename(columns={'Date': 'observation_date', 'Close': 'VIXCLS'})

# Ensure observation_date is datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Sort by observation_date
df = df.sort_values('observation_date').reset_index(drop=True)

# Compute indicators: Min of last 20 days and Max of last 10 days for VIX (shifted to exclude current day)
df['VIX_20D_Min'] = df['VIXCLS'].rolling(window=20).min().shift(1)
df['VIX_10D_Max'] = df['VIXCLS'].rolling(window=10).max().shift(1)

# Save to indicators folder
output_file = os.path.join(indicators_dir, 'VIX.csv')
df.to_csv(output_file, index=False)

print(f"Indicators computed and saved to {output_file}")