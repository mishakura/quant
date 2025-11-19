import pandas as pd
import os

# Paths
base_dir = os.path.dirname(__file__)
indicators_dir = os.path.join(base_dir, 'indicators')
csv_file = os.path.join(indicators_dir, 'vix_curve_indicators.csv')

# Load the CSV
df = pd.read_csv(csv_file)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Generate signals based on new rules: Entry when VIX_Price <= 13, Exit when VIX_Price >= 40
in_position = False
signals = []

for index, row in df.iterrows():
    vix_price = row['VIX_Price']
    
    if not in_position and vix_price <= 13:
        signal = 1
        in_position = True
    elif in_position and vix_price >= 40:
        signal = 0
        in_position = False
    else:
        signal = 1 if in_position else 0
    
    signals.append(signal)

df['Signal'] = signals

# Output to a new CSV in indicators folder
output_file = os.path.join(indicators_dir, 'trading_signals.csv')
df.to_csv(output_file)

print(f"Trading signals generated and saved to {output_file}")