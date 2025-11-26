import pandas as pd
import numpy as np

# Load SPX data
spx_df = pd.read_csv('SPX.csv')
spx_df['Date'] = pd.to_datetime(spx_df['Date'])
spx_df.set_index('Date', inplace=True)

# Load VVIX data
vvix_df = pd.read_csv('VVIX.csv')
vvix_df['Date'] = pd.to_datetime(vvix_df['Date'])
vvix_df.set_index('Date', inplace=True)

# Merge data on Date using outer join to include all dates from both CSVs
df = pd.merge(spx_df[['Close']], vvix_df[['Close']], left_index=True, right_index=True, suffixes=('_SPX', '_VVIX'), how='outer')

# Sort by date to ensure chronological order
df.sort_index(inplace=True)

# Fill NaN values with forward-fill, then backward-fill for any leading NaNs
df = df.fillna(method='ffill').fillna(method='bfill').dropna()

# Debug: Print data shapes and dates
print(f"SPX data shape: {spx_df.shape}")
print(f"VVIX data shape: {vvix_df.shape}")
print(f"Merged df shape: {df.shape}")
if len(df) > 0:
    print(f"First few dates: {df.index[:5].tolist()}")
    print(f"Last few dates: {df.index[-5:].tolist()}")
else:
    print("No data after merge and cleaning.")

# Calculate daily returns for SPX (assuming daily data)
df['Return'] = df['Close_SPX'].pct_change()

# Calculate 50-day EMA for SPX and VVIX (adjusted for daily data)
span = 50
df['EMA_SPX'] = df['Close_SPX'].ewm(span=span, adjust=False).mean()
df['EMA_VVIX'] = df['Close_VVIX'].ewm(span=span, adjust=False).mean()

# Debug: Print EMA values
if len(df) > 0:
    print(f"EMA_SPX first 5: {df['EMA_SPX'].head().tolist()}")
    print(f"EMA_VVIX first 5: {df['EMA_VVIX'].head().tolist()}")

# Initialize portfolio
initial_capital = 10000
portfolio_value = initial_capital
portfolio_returns = []
position = 0  # 0: cash, -1: short SPX
prev_position = 0
trade_count = 0
current_entry_price = np.nan

# Prepare a list to collect debug data
debug_data = []

# Loop over all periods to include full data in CSV, starting from 0
for i in range(len(df)):
    # Use today's prices and yesterday's EMAs for conditions (if available)
    spx_close = df['Close_SPX'].iloc[i]
    spx_ema = df['EMA_SPX'].iloc[i-1] if i >= span else np.nan
    vvix = df['Close_VVIX'].iloc[i]
    vvix_ema = df['EMA_VVIX'].iloc[i-1] if i >= span else np.nan
    
    entry_condition = (vvix > vvix_ema and spx_close < spx_ema) if i >= span else False
    
    # Debug print for first few iterations
    if i < span + 5:
        print(f"i={i}, date={df.index[i]}, spx_close={spx_close:.2f}, spx_ema={spx_ema:.2f}, vvix={vvix:.2f}, vvix_ema={vvix_ema:.2f}, condition={entry_condition}")
    
    # Entry: if condition and not already short, enter short
    if entry_condition and position == 0:
        position = -1
        current_entry_price = spx_close
        print(f"Entering short at {df.index[i]}")
    
    # Exit: if not condition and currently short, exit to cash
    elif not entry_condition and position == -1:
        position = 0
        print(f"Exiting short at {df.index[i]}")
    
    # Count trades
    if position == -1 and prev_position == 0:
        trade_count += 1
    
    prev_position = position
    
    # Apply return based on current position for the period
    daily_portfolio_return = position * df['Return'].iloc[i] if not np.isnan(df['Return'].iloc[i]) else 0
    
    portfolio_value *= (1 + daily_portfolio_return)
    portfolio_returns.append(daily_portfolio_return)
    
    # Calculate unrealized trade return
    unrealized_trade_return = - (spx_close - current_entry_price) / current_entry_price if not np.isnan(current_entry_price) else 0
    
    # Reset entry price on exit
    if position == 0:
        current_entry_price = np.nan
    
    # Collect debug data with all indicators, signals, and trade details
    debug_data.append({
        'Date': df.index[i],
        'SPX_Close': spx_close,
        'SPX_Return': df['Return'].iloc[i],
        'EMA_SPX': df['EMA_SPX'].iloc[i],
        'SPX_EMA_Yesterday': spx_ema,
        'VVIX': vvix,
        'EMA_VVIX': df['EMA_VVIX'].iloc[i],
        'VVIX_EMA_Yesterday': vvix_ema,
        'Entry_Condition': entry_condition,
        'Position': position,
        'Trade_Entered': 1 if position == -1 and prev_position == 0 else 0,
        'Trade_Exited': 1 if position == 0 and prev_position == -1 else 0,
        'Entry_Price': current_entry_price,
        'Unrealized_Trade_Return': unrealized_trade_return,
        'Daily_Portfolio_Return': daily_portfolio_return,
        'Portfolio_Value': portfolio_value
    })

# Create DataFrame and save to CSV with full data and backtest details
debug_df = pd.DataFrame(debug_data)
debug_df.to_csv('full_data_with_indicators.csv', index=False)

# Debug: Print trade count
print(f"Number of trades: {trade_count}")

# Calculate stats (adjusted for daily data, assuming 252 trading days per year)
if len(portfolio_returns) > 0:
    total_return = (portfolio_value - initial_capital) / initial_capital
    periods_per_year = 252  # Daily trading data
    years = len(portfolio_returns) / periods_per_year
    if years < 1:
        annualized_return = total_return  # For short periods, use total return as annualized
    else:
        annualized_return = (1 + total_return) ** (periods_per_year / len(portfolio_returns)) - 1
    volatility = np.std(portfolio_returns) * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

    # Max drawdown
    cumulative = np.cumprod(1 + np.array(portfolio_returns))
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
else:
    total_return = 0
    annualized_return = 0
    volatility = 0
    sharpe_ratio = 0
    max_drawdown = 0

# Output stats
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value: ${portfolio_value:.2f}")
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Volatility (Annualized): {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")