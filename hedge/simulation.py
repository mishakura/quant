import pandas as pd
import os
import yfinance as yf

# Paths
base_dir = os.path.dirname(__file__)
indicators_dir = os.path.join(base_dir, 'indicators')
signals_file = os.path.join(indicators_dir, 'trading_signals.csv')

# Load the signals CSV
df = pd.read_csv(signals_file)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Get VXX close prices using yfinance starting from 1930
start_date = pd.to_datetime('1930-01-01')
end_date = df.index.max() + pd.Timedelta(days=1)  # Include the end date
vxx_data = yf.download('UVXY', start=start_date, end=end_date)['Close']

# Find common dates between df and vxx_data
common_dates = df.index.intersection(vxx_data.index)

# Filter df to only include common dates
df = df.loc[common_dates]

# Add VXX close prices to df
df['VXX_Close'] = vxx_data.loc[common_dates]

# Initialize simulation variables
initial_capital = 10000
capital = initial_capital
position = 0  # Number of units held
position_size_pct = 0.10  # 10% position sizing

# Lists to store results
portfolio_values = []
capitals = []
positions = []

# Simulate trading using VXX_Close
for index, row in df.iterrows():
    price = row['VXX_Close']  # Use VXX close price for entry/exit
    signal = row['Signal']
    
    if signal == 1 and position == 0:
        # Enter long position with 10% of capital
        position_value = capital * position_size_pct
        position = position_value / price
        capital -= position_value
    elif signal == 0 and position > 0:
        # Exit position
        exit_value = position * price
        capital += exit_value
        position = 0
    
    # Calculate current portfolio value
    portfolio_value = capital + (position * price)
    
    # Store results
    capitals.append(capital)
    positions.append(position)
    portfolio_values.append(portfolio_value)

# Add results to DataFrame
df['Capital'] = capitals
df['Position'] = positions
df['Portfolio_Value'] = portfolio_values

# Output to a new CSV in hedge folder
output_file = os.path.join(base_dir, 'trading_simulation.csv')
df.to_csv(output_file)

print(f"Trading simulation completed and saved to {output_file}")
print(f"Initial Capital: {initial_capital}")
print(f"Final Portfolio Value: {portfolio_values[-1] if portfolio_values else initial_capital}")