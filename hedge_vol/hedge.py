import yfinance as yf
import pandas as pd
import numpy as np

# Define tickers
tickers = ['^VIX', '^VIX3M', 'SPY', 'VXX', 'SH']
start_date = '1990-01-01'  # Extended to get more data using VIX before VXX
end_date = '2025-11-25'

# Fetch data separately to avoid misalignment or missing data issues
data_frames = []
for ticker in tickers:
    if ticker == '^VIX':
        # Fetch VIX with Open and Close
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)[['Open', 'Close']].rename(columns={'Open': 'VIX_open', 'Close': 'VIX_close'})
    elif ticker == 'VXX':
        # Read VXX data from CSV (columns: time,open,high,low,close,Weekly close,Monthly close,Quaterly close)
        df = pd.read_csv('VXX.csv')
        df['time'] = pd.to_datetime(df['time'])  # 'time' is already in date string format
        df.set_index('time', inplace=True)
        df = df[['open', 'close']].rename(columns={'open': 'VXX_open', 'close': 'VXX_close'})  # Select 'open' and 'close' columns
    elif ticker == 'SH':
        # Fetch SH with Open and Close
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)[['Open', 'Close']].rename(columns={'Open': 'SH_open', 'Close': 'SH_close'})
    else:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)['Close']
        df.rename(columns={'Close': ticker}, inplace=True)
    data_frames.append(df)

# Merge all data on date index (use 'outer' join to include all dates, filling missing with NaN)
data = pd.concat(data_frames, axis=1, join='outer')

# Rename columns for clarity (remove ^ prefix)
data.columns = ['VIX_open', 'VIX_close', 'VIX3M', 'SPY', 'VXX_open', 'VXX_close', 'SH_open', 'SH_close']

# Create asset columns: Use VXX from 2009 onwards, VIX before
data['asset_open'] = data['VXX_open'].fillna(data['VIX_open'])
data['asset_close'] = data['VXX_close'].fillna(data['VIX_close'])

# Drop rows where asset data is missing (should not happen after fillna, but just in case)
data.dropna(subset=['asset_open', 'asset_close'], inplace=True)

# Drop any other rows with NaN in critical columns (e.g., SPY for realized_vol), but keep as much data as possible
data.dropna(subset=['SPY', 'VIX_close'], inplace=True)  # VIX_close is needed for eVRP and signal

# Compute daily returns for SPY
data['SPY_returns'] = data['SPY'].pct_change()

# Compute realized volatility: annualized std of last 10 days returns (multiplied by 100 to match VIX scale in %)
# For daily volatility (not annualized), use the commented line instead
data['realized_vol'] = (data['SPY_returns'].rolling(window=10).std() * np.sqrt(252)) * 100
# data['realized_vol'] = data['SPY_returns'].rolling(window=10).std() * 100  # Daily volatility in %

# Compute eVRP = VIX - realized_vol
data['eVRP'] = data['VIX_close'] - data['realized_vol']

# Add debug column: VIX - VIX3M
data['VIX_minus_VIX3M'] = data['VIX_close'] - data['VIX3M']

# Define raw signal: if eVRP <= 0 and VIX > VIX3M, buy asset; else, hold cash
# If VIX3M is NaN, the condition VIX_close > VIX3M will be False
data['raw_signal'] = (data['eVRP'] <= 0) & (data['VIX_close'] > data['VIX3M'])

# Shift raw signal to avoid lookahead bias for buy signal (use previous day's raw signal)
data['buy_signal'] = data['raw_signal'].shift(1)

# Compute asset returns (close-to-close for buy-and-hold)
data['VXX_returns'] = data['asset_close'].pct_change()

# Initialize strategy returns, position, and prices for VXX/VIX strategy
data['strategy_returns'] = 0.0
data['position'] = 0
data['entry_price'] = np.nan
data['exit_price'] = np.nan
entry_price = None
entry_index = None
trade_returns = []
holding_days = []

# Initialize strategy returns, position, and prices for SH strategy
data['SH_strategy_returns'] = 0.0
data['SH_position'] = 0
data['SH_entry_price'] = np.nan
data['SH_exit_price'] = np.nan
SH_entry_price = None
SH_entry_index = None
SH_trade_returns = []
SH_holding_days = []

# Loop to simulate multi-day holds for both strategies
for i in range(1, len(data)):
    # VXX/VIX strategy
    if data['buy_signal'].iloc[i] and data['position'].iloc[i-1] == 0:  # Enter only if signal is True and not already in position
        data.loc[data.index[i], 'position'] = 1
        entry_price = data['asset_open'].iloc[i]
        entry_index = i
        data.loc[data.index[i], 'entry_price'] = entry_price
    elif data['position'].iloc[i-1] == 1 and not data['buy_signal'].iloc[i]:  # Exit if position was held and signal is now False
        if entry_price is not None and entry_index is not None:
            exit_price = data['asset_open'].iloc[i]
            ret = (exit_price / entry_price) - 1  # (exit / entry) - 1
            data.loc[data.index[i], 'strategy_returns'] = ret
            data.loc[data.index[i], 'exit_price'] = exit_price
            trade_returns.append(ret)
            holding_days.append(i - entry_index)
            entry_price = None
            entry_index = None
        data.loc[data.index[i], 'position'] = 0
    else:
        data.loc[data.index[i], 'position'] = data['position'].iloc[i-1]
    
    # SH strategy (same signal, but buy SH)
    if data['buy_signal'].iloc[i] and data['SH_position'].iloc[i-1] == 0:  # Enter only if signal is True and not already in position
        data.loc[data.index[i], 'SH_position'] = 1
        SH_entry_price = data['SH_open'].iloc[i]
        SH_entry_index = i
        data.loc[data.index[i], 'SH_entry_price'] = SH_entry_price
    elif data['SH_position'].iloc[i-1] == 1 and not data['buy_signal'].iloc[i]:  # Exit if position was held and signal is now False
        if SH_entry_price is not None and SH_entry_index is not None:
            SH_exit_price = data['SH_open'].iloc[i]
            ret = (SH_exit_price / SH_entry_price) - 1  # (exit / entry) - 1
            data.loc[data.index[i], 'SH_strategy_returns'] = ret
            data.loc[data.index[i], 'SH_exit_price'] = SH_exit_price
            SH_trade_returns.append(ret)
            SH_holding_days.append(i - SH_entry_index)
            SH_entry_price = None
            SH_entry_index = None
        data.loc[data.index[i], 'SH_position'] = 0
    else:
        data.loc[data.index[i], 'SH_position'] = data['SH_position'].iloc[i-1]

# Calculate additional stats for VXX/VIX strategy
if trade_returns:
    total_trades = len(trade_returns)
    win_rate = sum(1 for r in trade_returns if r > 0) / total_trades
    avg_return_per_trade = np.mean(trade_returns)
    stdev_returns = np.std(trade_returns)
    avg_holding_days = np.mean(holding_days)
    max_win = max(trade_returns)
    max_loss = min(trade_returns)
    profit_factor = sum(r for r in trade_returns if r > 0) / abs(sum(r for r in trade_returns if r < 0)) if any(r < 0 for r in trade_returns) else np.inf
else:
    total_trades = 0
    win_rate = 0
    avg_return_per_trade = 0
    stdev_returns = 0
    avg_holding_days = 0
    max_win = 0
    max_loss = 0
    profit_factor = 0

# Calculate additional stats for SH strategy
if SH_trade_returns:
    SH_total_trades = len(SH_trade_returns)
    SH_win_rate = sum(1 for r in SH_trade_returns if r > 0) / SH_total_trades
    SH_avg_return_per_trade = np.mean(SH_trade_returns)
    SH_stdev_returns = np.std(SH_trade_returns)
    SH_avg_holding_days = np.mean(SH_holding_days)
    SH_max_win = max(SH_trade_returns)
    SH_max_loss = min(SH_trade_returns)
    SH_profit_factor = sum(r for r in SH_trade_returns if r > 0) / abs(sum(r for r in SH_trade_returns if r < 0)) if any(r < 0 for r in SH_trade_returns) else np.inf
else:
    SH_total_trades = 0
    SH_win_rate = 0
    SH_avg_return_per_trade = 0
    SH_stdev_returns = 0
    SH_avg_holding_days = 0
    SH_max_win = 0
    SH_max_loss = 0
    SH_profit_factor = 0

# Debug: Print entry and exit prices
entries = data.dropna(subset=['entry_price'])
exits = data.dropna(subset=['exit_price'])
if not entries.empty:
    print("VXX/VIX Entry Prices (Open of Buy Day):")
    for date, row in entries.iterrows():
        print(f"{date.date()}: Entry = {row['entry_price']:.2f}")
if not exits.empty:
    print("VXX/VIX Exit Prices (Open of Sell Day):")
    for date, row in exits.iterrows():
        print(f"{date.date()}: Exit = {row['exit_price']:.2f}")

SH_entries = data.dropna(subset=['SH_entry_price'])
SH_exits = data.dropna(subset=['SH_exit_price'])
if not SH_entries.empty:
    print("SH Entry Prices (Open of Buy Day):")
    for date, row in SH_entries.iterrows():
        print(f"{date.date()}: Entry = {row['SH_entry_price']:.2f}")
if not SH_exits.empty:
    print("SH Exit Prices (Open of Sell Day):")
    for date, row in SH_exits.iterrows():
        print(f"{date.date()}: Exit = {row['SH_exit_price']:.2f}")

# Cumulative returns
data['cum_strategy'] = (1 + data['strategy_returns']).cumprod()
data['cum_buy_hold'] = (1 + data['VXX_returns']).cumprod()
data['SH_cum_strategy'] = (1 + data['SH_strategy_returns']).cumprod()

# Initial capital
initial_capital = 10000
data['portfolio_value'] = initial_capital * data['cum_strategy']
data['SH_portfolio_value'] = initial_capital * data['SH_cum_strategy']

# Performance metrics for VXX/VIX
total_return = data['cum_strategy'].iloc[-1] - 1
# Annualized return: assuming 252 trading days per year
num_days = len(data)
annualized_return = (data['cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
std_returns = data['strategy_returns'].std()
if std_returns == 0:
    sharpe_ratio = 0  # Avoid division by zero
else:
    sharpe_ratio = data['strategy_returns'].mean() / std_returns * np.sqrt(252)
max_drawdown = (data['cum_strategy'] / data['cum_strategy'].cummax() - 1).min()

print("=== VXX/VIX Strategy ===")
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Final Portfolio Value: ${data['portfolio_value'].iloc[-1]:.2f}")

# Performance metrics for SH
SH_total_return = data['SH_cum_strategy'].iloc[-1] - 1
SH_annualized_return = (data['SH_cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
SH_std_returns = data['SH_strategy_returns'].std()
if SH_std_returns == 0:
    SH_sharpe_ratio = 0
else:
    SH_sharpe_ratio = data['SH_strategy_returns'].mean() / SH_std_returns * np.sqrt(252)
SH_max_drawdown = (data['SH_cum_strategy'] / data['SH_cum_strategy'].cummax() - 1).min()

print("=== SH Strategy ===")
print(f"Total Return: {SH_total_return:.2%}")
print(f"Annualized Return: {SH_annualized_return:.2%}")
print(f"Sharpe Ratio: {SH_sharpe_ratio:.2f}")
print(f"Max Drawdown: {SH_max_drawdown:.2%}")
print(f"Final Portfolio Value: ${data['SH_portfolio_value'].iloc[-1]:.2f}")

# Debug: Check signals and positions
print(f"Number of days buy_signal was True: {data['buy_signal'].sum()}")
print(f"Number of days VXX/VIX position was held: {data['position'].sum()}")
print(f"Number of days SH position was held: {data['SH_position'].sum()}")

# Additional stats for VXX/VIX
print("=== VXX/VIX Trade Stats ===")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Average Return per Trade: {avg_return_per_trade:.2%}")
print(f"Std Dev of Trade Returns: {stdev_returns:.2%}")
print(f"Average Holding Days per Position: {avg_holding_days:.1f}")
print(f"Max Win: {max_win:.2%}")
print(f"Max Loss: {max_loss:.2%}")
print(f"Profit Factor: {profit_factor:.2f}")

# Additional stats for SH
print("=== SH Trade Stats ===")
print(f"Total Trades: {SH_total_trades}")
print(f"Win Rate: {SH_win_rate:.2%}")
print(f"Average Return per Trade: {SH_avg_return_per_trade:.2%}")
print(f"Std Dev of Trade Returns: {SH_stdev_returns:.2%}")
print(f"Average Holding Days per Position: {SH_avg_holding_days:.1f}")
print(f"Max Win: {SH_max_win:.2%}")
print(f"Max Loss: {SH_max_loss:.2%}")
print(f"Profit Factor: {SH_profit_factor:.2f}")

# Export data to CSV for debugging
data.to_csv('debug_data.csv')

print("Data exported to debug_data.csv")