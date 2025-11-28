import pandas as pd
import numpy as np
import os

# Load data
directory = os.path.dirname(__file__)
spx_df = pd.read_csv(os.path.join(directory, 'SPX.csv'), parse_dates=['Date'])
gc_df = pd.read_csv(os.path.join(directory, 'GC.csv'), parse_dates=['Date'])

# Sort by date
spx_df = spx_df.sort_values('Date').reset_index(drop=True)
gc_df = gc_df.sort_values('Date').reset_index(drop=True)

# Merge on date
df = pd.merge(spx_df[['Date', 'Close']], gc_df[['Date', 'Close']], on='Date', suffixes=('_SPX', '_GC'))
df = df.sort_values('Date').reset_index(drop=True)

# Calculate indicators WITHOUT shift - we'll use yesterday's values properly
df['lowest_low_100'] = df['Close_SPX'].rolling(window=100, min_periods=100).min()
df['highest_high_50'] = df['Close_SPX'].rolling(window=50, min_periods=50).max()

# Initialize positions
df['position'] = 'full_spx'  # 'full_spx' or 'hedged'
df['spx_allocation'] = 1.0
df['gc_allocation'] = 0.0
df['cash_allocation'] = 0.0

# Strategy logic - compare today's close to YESTERDAY's indicators
position = 'full_spx'
for i in range(100, len(df)):  # Start after 100 days for valid indicator
    prev_lowest_low = df.loc[i-1, 'lowest_low_100']
    prev_highest_high = df.loc[i-1, 'highest_high_50']
    today_close = df.loc[i, 'Close_SPX']
    
    if position == 'full_spx':
        # Check if we should hedge: today's close < yesterday's 100-day low
        if today_close < prev_lowest_low:
            position = 'hedged'
            df.loc[i, 'position'] = 'hedged'
            df.loc[i, 'spx_allocation'] = 0.5
            df.loc[i, 'gc_allocation'] = 0.25
            df.loc[i, 'cash_allocation'] = 0.25
        else:
            df.loc[i, 'position'] = 'full_spx'
            df.loc[i, 'spx_allocation'] = 1.0
            df.loc[i, 'gc_allocation'] = 0.0
            df.loc[i, 'cash_allocation'] = 0.0
    else:  # position == 'hedged'
        # Check if we should exit hedge: today's close > yesterday's 50-day high
        if today_close > prev_highest_high:
            position = 'full_spx'
            df.loc[i, 'position'] = 'full_spx'
            df.loc[i, 'spx_allocation'] = 1.0
            df.loc[i, 'gc_allocation'] = 0.0
            df.loc[i, 'cash_allocation'] = 0.0
        else:
            df.loc[i, 'position'] = 'hedged'
            df.loc[i, 'spx_allocation'] = 0.5
            df.loc[i, 'gc_allocation'] = 0.25
            df.loc[i, 'cash_allocation'] = 0.25

# Calculate daily returns
df['spx_return'] = df['Close_SPX'].pct_change()
df['gc_return'] = df['Close_GC'].pct_change()

# Calculate portfolio returns
df['strategy_return'] = (df['spx_allocation'] * df['spx_return'] + 
                          df['gc_allocation'] * df['gc_return'])
df['buy_hold_return'] = df['spx_return']

# Calculate cumulative returns
df['strategy_cumulative'] = (1 + df['strategy_return']).cumprod()
df['buy_hold_cumulative'] = (1 + df['buy_hold_return']).cumprod()

# Performance statistics
def calculate_stats(returns, name):
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    
    # Proper max drawdown calculation
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"\n{name} Performance:")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Annualized Return: {annualized_return*100:.2f}%")
    print(f"Annualized Volatility: {annualized_vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# Calculate statistics (excluding first 100 days for warm-up)
valid_data = df[100:].copy()
strategy_stats = calculate_stats(valid_data['strategy_return'].dropna(), "Dynamic Hedge Strategy")
buy_hold_stats = calculate_stats(valid_data['buy_hold_return'].dropna(), "Buy & Hold SPX")

# Print trades
hedged_days = (df[100:]['position'] == 'hedged').sum()
print(f"\n\nNumber of hedge activations: {hedged_days}")
print(f"Percentage of time hedged: {hedged_days / len(df[100:]) * 100:.2f}%")

# Save results
df.to_csv(os.path.join(directory, 'backtest_results.csv'), index=False)
print(f"\nResults saved to backtest_results.csv")