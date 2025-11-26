import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Filter data to start from 2006-07-17, when VIX3M data begins
data = data[data.index >= '2006-07-17']

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
data['SH_returns'] = data['SH_close'].pct_change()

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

# Initialize strategy returns, position, and prices for Ultra Hedging strategy
data['ultra_strategy_returns'] = 0.0
data['ultra_position'] = 0
data['ultra_entry_price_VXX'] = np.nan
data['ultra_entry_price_SH'] = np.nan
ultra_entry_price_VXX = None
ultra_entry_price_SH = None
ultra_entry_index = None
ultra_trade_returns = []
ultra_holding_days = []

# Initialize strategy returns, position, and prices for Pure VXX strategy
data['pure_vxx_strategy_returns'] = 0.0
data['pure_vxx_position'] = 0
data['pure_vxx_entry_price'] = np.nan
data['pure_vxx_exit_price'] = np.nan
pure_vxx_entry_price = None
pure_vxx_entry_index = None
pure_vxx_trade_returns = []
pure_vxx_holding_days = []

# Initialize lists to store trade details for each strategy
trade_details = []
SH_trade_details = []
ultra_trade_details = []
pure_vxx_trade_details = []

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
            ret = 0.1 * ((exit_price / entry_price) - 1)  # Scaled by 0.1 for VXX allocation
            data.loc[data.index[i], 'strategy_returns'] = ret  # Note: This is still only on exit day, but we'll override later
            data.loc[data.index[i], 'exit_price'] = exit_price
            trade_returns.append(ret)
            holding_days.append(i - entry_index)
            # Add trade details
            entry_date = data.index[entry_index]
            exit_date = data.index[i]
            trade_details.append({'Strategy': 'VXX/VIX', 'Entry_Date': entry_date, 'Exit_Date': exit_date, 'Return': ret})
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
            # Add trade details
            SH_entry_date = data.index[SH_entry_index]
            SH_exit_date = data.index[i]
            SH_trade_details.append({'Strategy': 'SH', 'Entry_Date': SH_entry_date, 'Exit_Date': SH_exit_date, 'Return': ret})
            SH_entry_price = None
            SH_entry_index = None
        data.loc[data.index[i], 'SH_position'] = 0
    else:
        data.loc[data.index[i], 'SH_position'] = data['SH_position'].iloc[i-1]
    
    # Ultra Hedging strategy
    if data['buy_signal'].iloc[i] and data['ultra_position'].iloc[i-1] == 0:  # Enter only if signal is True and not already in position
        data.loc[data.index[i], 'ultra_position'] = 1
        ultra_entry_price_VXX = data['asset_open'].iloc[i]
        ultra_entry_price_SH = data['SH_open'].iloc[i]
        ultra_entry_index = i
        data.loc[data.index[i], 'ultra_entry_price_VXX'] = ultra_entry_price_VXX
        data.loc[data.index[i], 'ultra_entry_price_SH'] = ultra_entry_price_SH
    elif data['ultra_position'].iloc[i-1] == 1 and not data['buy_signal'].iloc[i]:  # Exit if position was held and signal is now False
        if ultra_entry_price_VXX is not None and ultra_entry_price_SH is not None and ultra_entry_index is not None:
            exit_price_VXX = data['asset_open'].iloc[i]
            exit_price_SH = data['SH_open'].iloc[i]
            ret_VXX = 0.1 * ((exit_price_VXX / ultra_entry_price_VXX) - 1)
            ret_SH = 0.1 * ((exit_price_SH / ultra_entry_price_SH) - 1)
            ret = ret_VXX + ret_SH
            data.loc[data.index[i], 'ultra_strategy_returns'] = ret  # Note: This is still only on exit day, but we'll override later
            ultra_trade_returns.append(ret)
            ultra_holding_days.append(i - ultra_entry_index)
            # Add trade details
            ultra_entry_date = data.index[ultra_entry_index]
            ultra_exit_date = data.index[i]
            ultra_trade_details.append({'Strategy': 'Ultra Hedging', 'Entry_Date': ultra_entry_date, 'Exit_Date': ultra_exit_date, 'Return': ret})
            ultra_entry_price_VXX = None
            ultra_entry_price_SH = None
            ultra_entry_index = None
        data.loc[data.index[i], 'ultra_position'] = 0
    else:
        data.loc[data.index[i], 'ultra_position'] = data['ultra_position'].iloc[i-1]
    
    # Pure VXX strategy (100% VXX when signal, cash otherwise)
    if data['buy_signal'].iloc[i] and data['pure_vxx_position'].iloc[i-1] == 0:  # Enter only if signal is True and not already in position
        data.loc[data.index[i], 'pure_vxx_position'] = 1
        pure_vxx_entry_price = data['asset_open'].iloc[i]
        pure_vxx_entry_index = i
        data.loc[data.index[i], 'pure_vxx_entry_price'] = pure_vxx_entry_price
    elif data['pure_vxx_position'].iloc[i-1] == 1 and not data['buy_signal'].iloc[i]:  # Exit if position was held and signal is now False
        if pure_vxx_entry_price is not None and pure_vxx_entry_index is not None:
            pure_vxx_exit_price = data['asset_open'].iloc[i]
            ret = (pure_vxx_exit_price / pure_vxx_entry_price) - 1  # Full return on VXX
            data.loc[data.index[i], 'pure_vxx_strategy_returns'] = ret
            data.loc[data.index[i], 'pure_vxx_exit_price'] = pure_vxx_exit_price
            pure_vxx_trade_returns.append(ret)
            pure_vxx_holding_days.append(i - pure_vxx_entry_index)
            # Add trade details
            pure_vxx_entry_date = data.index[pure_vxx_entry_index]
            pure_vxx_exit_date = data.index[i]
            pure_vxx_trade_details.append({'Strategy': 'Pure VXX', 'Entry_Date': pure_vxx_entry_date, 'Exit_Date': pure_vxx_exit_date, 'Return': ret})
            pure_vxx_entry_price = None
            pure_vxx_entry_index = None
        data.loc[data.index[i], 'pure_vxx_position'] = 0
    else:
        data.loc[data.index[i], 'pure_vxx_position'] = data['pure_vxx_position'].iloc[i-1]

# After the loop, calculate daily strategy returns for VXX/VIX
data['allocation_VXX'] = 0.20 * data['position']
data['allocation_SPY'] = 0.80 * data['position'] + 1.0 * (1 - data['position'])
data['strategy_returns'] = data['allocation_SPY'] * data['SPY_returns'] + data['allocation_VXX'] * data['VXX_returns']

# After the loop, calculate daily strategy returns for Ultra Hedging
data['ultra_allocation_VXX'] = 0.10 * data['ultra_position']
data['ultra_allocation_SH'] = 0.10 * data['ultra_position']
data['ultra_allocation_SPY'] = 0.80 * data['ultra_position'] + 1.0 * (1 - data['ultra_position'])
data['ultra_strategy_returns'] = data['ultra_allocation_SPY'] * data['SPY_returns'] + data['ultra_allocation_VXX'] * data['VXX_returns'] + data['ultra_allocation_SH'] * data['SH_returns']

# After the loop, calculate daily strategy returns for Pure VXX
data['pure_vxx_strategy_returns'] = data['pure_vxx_position'] * data['VXX_returns']  # 100% VXX or 0

# Handle initial NaN in returns for proper cumulative calculation
data['strategy_returns'] = data['strategy_returns'].fillna(0)
data['SPY_returns'] = data['SPY_returns'].fillna(0)
data['VXX_returns'] = data['VXX_returns'].fillna(0)
data['SH_returns'] = data['SH_returns'].fillna(0)
data['ultra_strategy_returns'] = data['ultra_strategy_returns'].fillna(0)
data['pure_vxx_strategy_returns'] = data['pure_vxx_strategy_returns'].fillna(0)

# Recalculate cumulative returns after filling NaN
data['cum_strategy'] = (1 + data['strategy_returns']).cumprod()
data['SPY_cum'] = (1 + data['SPY_returns']).cumprod()
data['ultra_cum_strategy'] = (1 + data['ultra_strategy_returns']).cumprod()
data['pure_vxx_cum_strategy'] = (1 + data['pure_vxx_strategy_returns']).cumprod()

num_days = len(data)

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

# Calculate additional stats for Ultra Hedging strategy
if ultra_trade_returns:
    ultra_total_trades = len(ultra_trade_returns)
    ultra_win_rate = sum(1 for r in ultra_trade_returns if r > 0) / ultra_total_trades
    ultra_avg_return_per_trade = np.mean(ultra_trade_returns)
    ultra_stdev_returns = np.std(ultra_trade_returns)
    ultra_avg_holding_days = np.mean(ultra_holding_days)
    ultra_max_win = max(ultra_trade_returns)
    ultra_max_loss = min(ultra_trade_returns)
    ultra_profit_factor = sum(r for r in ultra_trade_returns if r > 0) / abs(sum(r for r in ultra_trade_returns if r < 0)) if any(r < 0 for r in ultra_trade_returns) else np.inf
else:
    ultra_total_trades = 0
    ultra_win_rate = 0
    ultra_avg_return_per_trade = 0
    ultra_stdev_returns = 0
    ultra_avg_holding_days = 0
    ultra_max_win = 0
    ultra_max_loss = 0
    ultra_profit_factor = 0

# Calculate additional stats for Pure VXX strategy
if pure_vxx_trade_returns:
    pure_vxx_total_trades = len(pure_vxx_trade_returns)
    pure_vxx_win_rate = sum(1 for r in pure_vxx_trade_returns if r > 0) / pure_vxx_total_trades
    pure_vxx_avg_return_per_trade = np.mean(pure_vxx_trade_returns)
    pure_vxx_stdev_returns = np.std(pure_vxx_trade_returns)
    pure_vxx_avg_holding_days = np.mean(pure_vxx_holding_days)
    pure_vxx_max_win = max(pure_vxx_trade_returns)
    pure_vxx_max_loss = min(pure_vxx_trade_returns)
    pure_vxx_profit_factor = sum(r for r in pure_vxx_trade_returns if r > 0) / abs(sum(r for r in pure_vxx_trade_returns if r < 0)) if any(r < 0 for r in pure_vxx_trade_returns) else np.inf
else:
    pure_vxx_total_trades = 0
    pure_vxx_win_rate = 0
    pure_vxx_avg_return_per_trade = 0
    pure_vxx_stdev_returns = 0
    pure_vxx_avg_holding_days = 0
    pure_vxx_max_win = 0
    pure_vxx_max_loss = 0
    pure_vxx_profit_factor = 0

# Additional stats for SPY Buy-and-Hold
SPY_daily_returns = data['SPY_returns']
SPY_std_dev = SPY_daily_returns.std()
SPY_skewness = SPY_daily_returns.skew()
SPY_kurtosis = SPY_daily_returns.kurt()
SPY_geometric_return = (data['SPY_cum'].iloc[-1]) ** (252 / num_days) - 1
SPY_drawdowns = (data['SPY_cum'] / data['SPY_cum'].cummax() - 1)
SPY_max_drawdown = SPY_drawdowns.min()
SPY_avg_drawdown = SPY_drawdowns.mean()
SPY_drawdown_std = SPY_drawdowns.std()

# Additional stats for VXX/VIX Strategy
VXX_daily_returns = data['strategy_returns']
VXX_std_dev = VXX_daily_returns.std()
VXX_skewness = VXX_daily_returns.skew()
VXX_kurtosis = VXX_daily_returns.kurt()
VXX_geometric_return = (data['cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
VXX_drawdowns = (data['cum_strategy'] / data['cum_strategy'].cummax() - 1)
VXX_max_drawdown = VXX_drawdowns.min()
VXX_avg_drawdown = VXX_drawdowns.mean()
VXX_drawdown_std = VXX_drawdowns.std()

# Additional stats for Ultra Hedging Strategy
ultra_daily_returns = data['ultra_strategy_returns']
ultra_std_dev = ultra_daily_returns.std()
ultra_skewness = ultra_daily_returns.skew()
ultra_kurtosis = ultra_daily_returns.kurt()
ultra_geometric_return = (data['ultra_cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
ultra_drawdowns = (data['ultra_cum_strategy'] / data['ultra_cum_strategy'].cummax() - 1)
ultra_max_drawdown = ultra_drawdowns.min()
ultra_avg_drawdown = ultra_drawdowns.mean()
ultra_drawdown_std = ultra_drawdowns.std()

# Additional stats for Pure VXX Strategy
pure_vxx_daily_returns = data['pure_vxx_strategy_returns']
pure_vxx_std_dev = pure_vxx_daily_returns.std()
pure_vxx_skewness = pure_vxx_daily_returns.skew()
pure_vxx_kurtosis = pure_vxx_daily_returns.kurt()
pure_vxx_geometric_return = (data['pure_vxx_cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
pure_vxx_drawdowns = (data['pure_vxx_cum_strategy'] / data['pure_vxx_cum_strategy'].cummax() - 1)
pure_vxx_max_drawdown = pure_vxx_drawdowns.min()
pure_vxx_avg_drawdown = pure_vxx_drawdowns.mean()
pure_vxx_drawdown_std = pure_vxx_drawdowns.std()

# Define performance metrics for stats_df
SPY_total_return = data['SPY_cum'].iloc[-1] - 1
SPY_annualized_return_calc = (data['SPY_cum'].iloc[-1]) ** (252 / num_days) - 1
SPY_std_returns = data['SPY_returns'].std()
SPY_sharpe_ratio = data['SPY_returns'].mean() / SPY_std_returns * np.sqrt(252) if SPY_std_returns != 0 else 0

total_return = data['cum_strategy'].iloc[-1] - 1
annualized_return_calc = (data['cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
std_returns = data['strategy_returns'].std()
sharpe_ratio = data['strategy_returns'].mean() / std_returns * np.sqrt(252) if std_returns != 0 else 0

ultra_total_return = data['ultra_cum_strategy'].iloc[-1] - 1
ultra_annualized_return_calc = (data['ultra_cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
ultra_std_returns = data['ultra_strategy_returns'].std()
ultra_sharpe_ratio = data['ultra_strategy_returns'].mean() / ultra_std_returns * np.sqrt(252) if ultra_std_returns != 0 else 0

pure_vxx_total_return = data['pure_vxx_cum_strategy'].iloc[-1] - 1
pure_vxx_annualized_return_calc = (data['pure_vxx_cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
pure_vxx_std_returns = data['pure_vxx_strategy_returns'].std()
pure_vxx_sharpe_ratio = data['pure_vxx_strategy_returns'].mean() / pure_vxx_std_returns * np.sqrt(252) if pure_vxx_std_returns != 0 else 0

# Resample to monthly for drawdowns and returns
monthly_SPY_cum = data['SPY_cum'].resample('M').last()
monthly_VXX_cum = data['cum_strategy'].resample('M').last()
monthly_ultra_cum = data['ultra_cum_strategy'].resample('M').last()
monthly_pure_vxx_cum = data['pure_vxx_cum_strategy'].resample('M').last()

# Monthly drawdowns
monthly_SPY_drawdowns = (monthly_SPY_cum / monthly_SPY_cum.cummax() - 1)
monthly_VXX_drawdowns = (monthly_VXX_cum / monthly_VXX_cum.cummax() - 1)
monthly_ultra_drawdowns = (monthly_ultra_cum / monthly_ultra_cum.cummax() - 1)
monthly_pure_vxx_drawdowns = (monthly_pure_vxx_cum / monthly_pure_vxx_cum.cummax() - 1)

# Update drawdown stats to monthly
SPY_max_drawdown = monthly_SPY_drawdowns.min()
SPY_avg_drawdown = monthly_SPY_drawdowns.mean()
SPY_drawdown_std = monthly_SPY_drawdowns.std()

VXX_max_drawdown = monthly_VXX_drawdowns.min()
VXX_avg_drawdown = monthly_VXX_drawdowns.mean()
VXX_drawdown_std = monthly_VXX_drawdowns.std()

ultra_max_drawdown = monthly_ultra_drawdowns.min()
ultra_avg_drawdown = monthly_ultra_drawdowns.mean()
ultra_drawdown_std = monthly_ultra_drawdowns.std()

pure_vxx_max_drawdown = monthly_pure_vxx_drawdowns.min()
pure_vxx_avg_drawdown = monthly_pure_vxx_drawdowns.mean()
pure_vxx_drawdown_std = monthly_pure_vxx_drawdowns.std()

# Annualize std dev
SPY_std_dev_annualized = SPY_std_dev * np.sqrt(252)
VXX_std_dev_annualized = VXX_std_dev * np.sqrt(252)
ultra_std_dev_annualized = ultra_std_dev * np.sqrt(252)
pure_vxx_std_dev_annualized = pure_vxx_std_dev * np.sqrt(252)

# Monthly returns
SPY_monthly_returns = data['SPY_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
VXX_monthly_returns = data['strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
ultra_monthly_returns = data['ultra_strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
pure_vxx_monthly_returns = data['pure_vxx_strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)

# Create DataFrame for monthly returns
monthly_returns_df = pd.DataFrame({
    'Date': SPY_monthly_returns.index,
    'SPY Monthly Return': SPY_monthly_returns.values,
    'VXX/VIX Monthly Return': VXX_monthly_returns.values,
    'Ultra Monthly Return': ultra_monthly_returns.values,
    'Pure VXX Monthly Return': pure_vxx_monthly_returns.values
})

# Export monthly returns to CSV
monthly_returns_df.to_csv('monthly_returns.csv', index=False)

# After all calculations, create a combined DataFrame for all trade returns
all_trade_details = trade_details + SH_trade_details + ultra_trade_details + pure_vxx_trade_details
trade_returns_df = pd.DataFrame(all_trade_details)

# Export to CSV
trade_returns_df.to_csv('all_trade_returns.csv', index=False)

print("Stats exported to strategy_stats.csv")
print("Drawdowns exported to strategy_drawdowns.csv")
print("All trade returns exported to all_trade_returns.csv")

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
data['SPY_cum'] = (1 + data['SPY_returns']).cumprod()
data['ultra_cum_strategy'] = (1 + data['ultra_strategy_returns']).cumprod()
data['pure_vxx_cum_strategy'] = (1 + data['pure_vxx_strategy_returns']).cumprod()

# Initial capital
initial_capital = 10000
data['portfolio_value'] = initial_capital * data['cum_strategy']
data['SH_portfolio_value'] = initial_capital * data['SH_cum_strategy']
data['SPY_portfolio_value'] = initial_capital * data['SPY_cum']
data['ultra_portfolio_value'] = initial_capital * data['ultra_cum_strategy']
data['pure_vxx_portfolio_value'] = initial_capital * data['pure_vxx_cum_strategy']

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

# Performance metrics for Ultra Hedging
ultra_total_return = data['ultra_cum_strategy'].iloc[-1] - 1
ultra_annualized_return = (data['ultra_cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
ultra_std_returns = data['ultra_strategy_returns'].std()
if ultra_std_returns == 0:
    ultra_sharpe_ratio = 0
else:
    ultra_sharpe_ratio = data['ultra_strategy_returns'].mean() / ultra_std_returns * np.sqrt(252)
ultra_max_drawdown = (data['ultra_cum_strategy'] / data['ultra_cum_strategy'].cummax() - 1).min()

print("=== Ultra Hedging Strategy ===")
print(f"Total Return: {ultra_total_return:.2%}")
print(f"Annualized Return: {ultra_annualized_return:.2%}")
print(f"Sharpe Ratio: {ultra_sharpe_ratio:.2f}")
print(f"Max Drawdown: {ultra_max_drawdown:.2%}")
print(f"Final Portfolio Value: ${data['ultra_portfolio_value'].iloc[-1]:.2f}")

# Performance metrics for Pure VXX
pure_vxx_total_return = data['pure_vxx_cum_strategy'].iloc[-1] - 1
pure_vxx_annualized_return = (data['pure_vxx_cum_strategy'].iloc[-1]) ** (252 / num_days) - 1
pure_vxx_std_returns = data['pure_vxx_strategy_returns'].std()
if pure_vxx_std_returns == 0:
    pure_vxx_sharpe_ratio = 0
else:
    pure_vxx_sharpe_ratio = data['pure_vxx_strategy_returns'].mean() / pure_vxx_std_returns * np.sqrt(252)
pure_vxx_max_drawdown = (data['pure_vxx_cum_strategy'] / data['pure_vxx_cum_strategy'].cummax() - 1).min()

print("=== Pure VXX Strategy ===")
print(f"Total Return: {pure_vxx_total_return:.2%}")
print(f"Annualized Return: {pure_vxx_annualized_return:.2%}")
print(f"Sharpe Ratio: {pure_vxx_sharpe_ratio:.2f}")
print(f"Max Drawdown: {pure_vxx_max_drawdown:.2%}")
print(f"Final Portfolio Value: ${data['pure_vxx_portfolio_value'].iloc[-1]:.2f}")

# Performance metrics for SPY Buy-and-Hold
SPY_total_return = data['SPY_cum'].iloc[-1] - 1
SPY_annualized_return = (data['SPY_cum'].iloc[-1]) ** (252 / num_days) - 1
SPY_std_returns = data['SPY_returns'].std()
if SPY_std_returns == 0:
    SPY_sharpe_ratio = 0
else:
    SPY_sharpe_ratio = data['SPY_returns'].mean() / SPY_std_returns * np.sqrt(252)
SPY_drawdowns = (data['SPY_cum'] / data['SPY_cum'].cummax() - 1)
SPY_max_drawdown = SPY_drawdowns.min()
SPY_avg_drawdown = SPY_drawdowns.mean()
SPY_drawdown_std = SPY_drawdowns.std()

print("=== SPY Buy-and-Hold Strategy ===")
print(f"Total Return: {SPY_total_return:.2%}")
print(f"Annualized Return: {SPY_annualized_return:.2%}")
print(f"Sharpe Ratio: {SPY_sharpe_ratio:.2f}")
print(f"Max Drawdown: {SPY_max_drawdown:.2%}")
print(f"Final Portfolio Value: ${data['SPY_portfolio_value'].iloc[-1]:.2f}")

# Debug: Check signals and positions
print(f"Number of days buy_signal was True: {data['buy_signal'].sum()}")
print(f"Number of days VXX/VIX position was held: {data['position'].sum()}")
print(f"Number of days SH position was held: {data['SH_position'].sum()}")
print(f"Number of days Ultra position was held: {data['ultra_position'].sum()}")
print(f"Number of days Pure VXX position was held: {data['pure_vxx_position'].sum()}")

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

# Additional stats for Ultra Hedging
print("=== Ultra Hedging Trade Stats ===")
print(f"Total Trades: {ultra_total_trades}")
print(f"Win Rate: {ultra_win_rate:.2%}")
print(f"Average Return per Trade: {ultra_avg_return_per_trade:.2%}")
print(f"Std Dev of Trade Returns: {ultra_stdev_returns:.2%}")
print(f"Average Holding Days per Position: {ultra_avg_holding_days:.1f}")
print(f"Max Win: {ultra_max_win:.2%}")
print(f"Max Loss: {ultra_max_loss:.2%}")
print(f"Profit Factor: {ultra_profit_factor:.2f}")

# Additional stats for Pure VXX
print("=== Pure VXX Trade Stats ===")
print(f"Total Trades: {pure_vxx_total_trades}")
print(f"Win Rate: {pure_vxx_win_rate:.2%}")
print(f"Average Return per Trade: {pure_vxx_avg_return_per_trade:.2%}")
print(f"Std Dev of Trade Returns: {pure_vxx_stdev_returns:.2%}")
print(f"Average Holding Days per Position: {pure_vxx_avg_holding_days:.1f}")
print(f"Max Win: {pure_vxx_max_win:.2%}")
print(f"Max Loss: {pure_vxx_max_loss:.2%}")
print(f"Profit Factor: {pure_vxx_profit_factor:.2f}")

# Export data to CSV for debugging
data.to_csv('debug_data.csv')

print("Data exported to debug_data.csv")

# Plot comparison of VXX/VIX Strategy, Ultra Hedging Strategy, Pure VXX Strategy, and SPY Buy-and-Hold
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['cum_strategy'], label='VXX/VIX Strategy', linewidth=2)
plt.plot(data.index, data['ultra_cum_strategy'], label='Ultra Hedging Strategy', linewidth=2)
plt.plot(data.index, data['pure_vxx_cum_strategy'], label='Pure VXX Strategy', linewidth=2)
plt.plot(data.index, data['SPY_cum'], label='SPY Buy-and-Hold', linewidth=2)
plt.title('Cumulative Returns: Strategies vs SPY Buy-and-Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd

# Load the CSV (adjust path if needed)
df = pd.read_csv('strategy_drawdowns.csv')

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Function to compute within-month max drawdown
def compute_within_month_max_drawdown(series):
    # Group by year-month
    monthly_groups = series.groupby(series.index.to_period('M'))
    results = []
    for period, group in monthly_groups:
        # Compute cumulative returns within the month (starting from 1.0)
        monthly_cum = (1 + group).cumprod()
        # Find the max cumulative return within the month
        monthly_max = monthly_cum.cummax()
        # Compute drawdowns within the month
        monthly_drawdowns = monthly_cum / monthly_max - 1
        # Get the max drawdown (most negative) in the month
        max_drawdown = monthly_drawdowns.min()
        results.append({'Year-Month': period, 'Max Drawdown': max_drawdown})
    return pd.DataFrame(results)

# Process SPY Drawdowns (within-month)
SPY_drawdowns = df.set_index('Date')['SPY Drawdown']
SPY_monthly_max_drawdowns = compute_within_month_max_drawdown(SPY_drawdowns)
SPY_monthly_max_drawdowns.rename(columns={'Max Drawdown': 'SPY Monthly Max Drawdown'}, inplace=True)

# Process VXX/VIX Drawdowns (within-month)
VXX_drawdowns = df.set_index('Date')['VXX/VIX Drawdown']
VXX_monthly_max_drawdowns = compute_within_month_max_drawdown(VXX_drawdowns)
VXX_monthly_max_drawdowns.rename(columns={'Max Drawdown': 'VXX/VIX Monthly Max Drawdown'}, inplace=True)

# Process Ultra Drawdowns (within-month)
ultra_drawdowns = df.set_index('Date')['Ultra Drawdown']
ultra_monthly_max_drawdowns = compute_within_month_max_drawdown(ultra_drawdowns)
ultra_monthly_max_drawdowns.rename(columns={'Max Drawdown': 'Ultra Monthly Max Drawdown'}, inplace=True)

# Process Pure VXX Drawdowns (within-month)
pure_vxx_drawdowns = df.set_index('Date')['Pure VXX Drawdown']
pure_vxx_monthly_max_drawdowns = compute_within_month_max_drawdown(pure_vxx_drawdowns)
pure_vxx_monthly_max_drawdowns.rename(columns={'Max Drawdown': 'Pure VXX Monthly Max Drawdown'}, inplace=True)

# Merge into one DataFrame
monthly_drawdowns = pd.merge(SPY_monthly_max_drawdowns, VXX_monthly_max_drawdowns, on='Year-Month', how='outer')
monthly_drawdowns = pd.merge(monthly_drawdowns, ultra_monthly_max_drawdowns, on='Year-Month', how='outer')
monthly_drawdowns = pd.merge(monthly_drawdowns, pure_vxx_monthly_max_drawdowns, on='Year-Month', how='outer')

# Save the result to a new CSV
monthly_drawdowns.to_csv('monthly_drawdowns_only.csv', index=False)

# Optional: Print a preview
print(monthly_drawdowns.head())