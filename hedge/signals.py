import pandas as pd
import os

# Paths
base_dir = os.path.dirname(__file__)
indicators_dir = os.path.join(base_dir, 'indicators')
csv_file = os.path.join(indicators_dir, 'vxx_indicators.csv')

# Load the CSV
df = pd.read_csv(csv_file)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate EMA8 and EMA32 on VXX_Price
df['EMA8'] = df['VXX_Price'].ewm(span=50).mean()
df['EMA32'] = df['VXX_Price'].ewm(span=100).mean()

# Generate signals: 1 when EMA8 > EMA32 (go long), 0 otherwise (close trade)
df['Signal'] = (df['EMA8'] > df['EMA32']).astype(int)

# Output to a new CSV in indicators folder
output_file = os.path.join(indicators_dir, 'trading_signals.csv')
df.to_csv(output_file)

print(f"Trading signals generated and saved to {output_file}")