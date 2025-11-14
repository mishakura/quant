import pandas as pd
import os
import numpy as np

# Paths
indicators_dir = os.path.join(os.path.dirname(__file__), 'indicators')

# Load the VIX indicators data
vix_file = os.path.join(indicators_dir, 'VIX.csv')
df = pd.read_csv(vix_file)

# Compute signal: 
# Enter when VIX close price is below the last 20 days min (VIX_20D_Min is min of previous 20 days)
# Close the trade when VIX close price is above the last 10 max days (VIX_10D_Max is max of previous 10 days)
df['Signal'] = 0
in_trade = False
for i in range(len(df)):
    if not in_trade and pd.notna(df.loc[i, 'VIX_20D_Min']) and df.loc[i, 'VIXCLS'] < df.loc[i, 'VIX_20D_Min']:
        in_trade = True
    if in_trade and pd.notna(df.loc[i, 'VIX_10D_Max']) and df.loc[i, 'VIXCLS'] > df.loc[i, 'VIX_10D_Max']:
        in_trade = False
    df.loc[i, 'Signal'] = 1 if in_trade else 0

# Save back to the same file
df.to_csv(vix_file, index=False)

print(f"Signals computed and added to {vix_file}")