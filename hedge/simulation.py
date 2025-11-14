import pandas as pd
import os
import numpy as np

# Paths
indicators_dir = os.path.join(os.path.dirname(__file__), 'indicators')

# Load the VIX indicators data
vix_file = os.path.join(indicators_dir, 'VIX.csv')
df = pd.read_csv(vix_file)

# Identify trades: assign Trade_ID for consecutive Signal == 1
df['Trade_ID'] = (df['Signal'] != df['Signal'].shift()).cumsum()

# Compute final PnL for each trade: exit_price / entry_price - 1
trade_pnl = df[df['Signal'] == 1].groupby('Trade_ID').agg(
    entry_price=('VIXCLS', 'first'),
    exit_price=('VIXCLS', 'last')
)
trade_pnl['Trade_PnL'] = trade_pnl['exit_price'] / trade_pnl['entry_price'] - 1

# Drop existing Trade_PnL column if it exists
df.drop('Trade_PnL', axis=1, errors='ignore', inplace=True)

# Merge back to df
df = df.merge(trade_pnl[['Trade_PnL']], left_on='Trade_ID', right_index=True, how='left')

# Identify exit days: where Signal == 1 and next Signal == 0 or last row
df['Exit_Day'] = (df['Signal'] == 1) & ((df['Signal'].shift(-1) == 0) | (df.index == len(df) - 1))

# Trade return: Trade_PnL on exit days, else 0
df['Trade_Return'] = np.where(df['Exit_Day'], df['Trade_PnL'], 0)

# Fill NaN in Trade_Return with 0 to avoid NaN in capital
df['Trade_Return'] = df['Trade_Return'].fillna(0)

# Compute unrealized PnL daily
entry_prices = df[df['Signal'] == 1].groupby('Trade_ID')['VIXCLS'].first()
df['Entry_Price'] = df['Trade_ID'].map(entry_prices)
df['Unrealized_PnL'] = np.where(df['Signal'] == 1, df['VIXCLS'] / df['Entry_Price'] - 1, 0)

# Simulate capital starting from 10,000 USD
initial_capital = 10000
df['Capital'] = initial_capital * (1 + df['Trade_Return']).cumprod()

# Drop temporary columns
df.drop(['Trade_ID', 'Entry_Price'], axis=1, inplace=True)

# Save back to the same file
df.to_csv(vix_file, index=False)

print(f"Simulation completed and added to {vix_file}")