import pandas as pd
import os

# Path to the VIX file
data_dir = os.path.join(os.path.dirname(__file__), 'data')
vix_file = os.path.join(data_dir, 'VIX.csv')

# Load the VIX data
df = pd.read_csv(vix_file)

# 1. Handle missing values: Drop rows with NaN in observation_date or VIXCLS
df = df.dropna(subset=['observation_date', 'VIXCLS'])

# 2. Remove duplicates based on observation_date
df = df.drop_duplicates(subset=['observation_date'])

# 3. Convert data types
df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
df['VIXCLS'] = pd.to_numeric(df['VIXCLS'], errors='coerce')

# Drop rows where conversion failed
df = df.dropna(subset=['observation_date', 'VIXCLS'])

# 4. Sort by observation_date
df = df.sort_values('observation_date').reset_index(drop=True)

# 5. Handle invalid values: Ensure VIXCLS is positive
df = df[df['VIXCLS'] > 0]

# Save back to the file
df.to_csv(vix_file, index=False)

print(f"VIX data cleaned and saved to {vix_file}")