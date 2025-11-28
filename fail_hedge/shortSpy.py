import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Load SPX data
spx_df = pd.read_csv('SPX.csv')
spx_df['Date'] = pd.to_datetime(spx_df['Date'])
spx_df.set_index('Date', inplace=True)

# Load VVIX data
vvix_df = pd.read_csv('VVIX.csv')
vvix_df['Date'] = pd.to_datetime(vvix_df['Date'])
vvix_df.set_index('Date', inplace=True)

# Load VIX data
vix_df = pd.read_csv('VIX.csv')
vix_df['Date'] = pd.to_datetime(vix_df['Date'])
vix_df.set_index('Date', inplace=True)

# Load SH data from yfinance
sh_df = yf.download('SH', start='2006-06-21', end='2025-11-28')
sh_df = sh_df[['Close', 'Open']].copy()
sh_df.columns = ['Close_SH', 'Open_SH']
sh_df.index = pd.to_datetime(sh_df.index)

# Load QQQ data from yfinance
qqq_df = yf.download('QQQ', start='2006-06-21', end='2025-11-28')
qqq_df = qqq_df[['Close', 'Open']].copy()
qqq_df.columns = ['Close_QQQ', 'Open_QQQ']
qqq_df.index = pd.to_datetime(qqq_df.index)

# Load PSQ data from yfinance
psq_df = yf.download('PSQ', start='2006-06-21', end='2025-11-28')
psq_df = psq_df[['Close', 'Open']].copy()
psq_df.columns = ['Close_PSQ', 'Open_PSQ']
psq_df.index = pd.to_datetime(psq_df.index)

# Merge data on Date using outer join to include all dates from both CSVs
df = pd.merge(spx_df[['Close', 'Open']], vvix_df[['Close', 'Open']], left_index=True, right_index=True, suffixes=('_SPX', '_VVIX'), how='outer')
df = pd.merge(df, vix_df[['Close', 'Open']], left_index=True, right_index=True, how='outer')
df = df.rename(columns={'Close': 'Close_VIX', 'Open': 'Open_VIX'})
df = pd.merge(df, sh_df[['Close_SH', 'Open_SH']], left_index=True, right_index=True, how='outer')
df = pd.merge(df, qqq_df[['Close_QQQ', 'Open_QQQ']], left_index=True, right_index=True, how='outer')
df = pd.merge(df, psq_df[['Close_PSQ', 'Open_PSQ']], left_index=True, right_index=True, how='outer')

# Sort by date to ensure chronological order
df.sort_index(inplace=True)

# Fill NaN values with forward-fill, then backward-fill for any leading NaNs
df = df.fillna(method='ffill').fillna(method='bfill').dropna()

# Truncate data to start from 2006-06-21
df = df[df.index >= '2006-06-21']

# Debug: Print data shapes and dates
print(f"SPX data shape: {spx_df.shape}")
print(f"VVIX data shape: {vvix_df.shape}")
print(f"VIX data shape: {vix_df.shape}")
print(f"SH data shape: {sh_df.shape}")
print(f"QQQ data shape: {qqq_df.shape}")
print(f"PSQ data shape: {psq_df.shape}")
print(f"Merged df shape: {df.shape}")
if len(df) > 0:
    print(f"First few dates: {df.index[:5].tolist()}")
    print(f"Last few dates: {df.index[-5:].tolist()}")
else:
    print("No data after merge and cleaning.")

# Calculate daily returns for SPX, SH, QQQ, and PSQ (assuming daily data)
df['Return_SPX'] = df['Close_SPX'].pct_change()
df['Return_SH'] = df['Close_SH'].pct_change()
df['Return_QQQ'] = df['Close_QQQ'].pct_change()
df['Return_PSQ'] = df['Close_PSQ'].pct_change()

# Debug: Check returns
print("SH Return stats:")
print(df['Return_SH'].describe())
print("SPX Return stats:")
print(df['Return_SPX'].describe())
print("QQQ Return stats:")
print(df['Return_QQQ'].describe())
print("PSQ Return stats:")
print(df['Return_PSQ'].describe())
print("Correlation between SPX and SH returns:")
print(df[['Return_SPX', 'Return_SH']].corr())
print("Correlation between QQQ and PSQ returns:")
print(df[['Return_QQQ', 'Return_PSQ']].corr())

# Calculate EMAs for SPX, VVIX, VIX, and QQQ with separate spans
span_spx = 10
span_vvix = 100
df['EMA_SPX'] = df['Close_SPX'].ewm(span=span_spx, adjust=False).mean()
df['EMA_VVIX'] = df['Close_VVIX'].ewm(span=span_vvix, adjust=False).mean()
df['EMA_VIX'] = df['Close_VIX'].ewm(span=span_vvix, adjust=False).mean()  # Same span as VVIX
df['EMA_QQQ'] = df['Close_QQQ'].ewm(span=span_spx, adjust=False).mean()  # Same span as SPX

# Set span to the maximum of the two for the loop condition
span = max(span_spx, span_vvix)

# Debug: Print EMA values
if len(df) > 0:
    print(f"EMA_SPX first 5: {df['EMA_SPX'].head().tolist()}")
    print(f"EMA_VVIX first 5: {df['EMA_VVIX'].head().tolist()}")
    print(f"EMA_VIX first 5: {df['EMA_VIX'].head().tolist()}")
    print(f"EMA_QQQ first 5: {df['EMA_QQQ'].head().tolist()}")

# Initialize portfolios for all strategies
initial_capital = 100000
portfolio_value_main = initial_capital  # Short SPX strategy
portfolio_value_sh = initial_capital    # Long SH strategy
portfolio_value_vix_sh = initial_capital  # VIX-SH strategy
portfolio_value_long_spx = initial_capital  # Long Only SPX strategy
portfolio_value_dynamic = initial_capital  # Dynamic SH Strategy
portfolio_value_dynamic_psq = initial_capital  # Dynamic PSQ Strategy
portfolio_value_psq_alone = initial_capital  # PSQ Alone Strategy
portfolio_returns_main = []
portfolio_returns_sh = []
portfolio_returns_vix_sh = []
portfolio_returns_long_spx = []  # For long only, collect daily returns
portfolio_returns_dynamic = []  # For dynamic, collect daily returns
portfolio_returns_dynamic_psq = []  # For dynamic PSQ, collect daily returns
portfolio_returns_psq_alone = []  # For PSQ alone, collect trade returns
position_main = 0  # 0: cash, -1: short SPX
position_sh = 0    # 0: cash, 1: long SH
position_vix_sh = 0  # 0: cash, 1: long SH
position_long_spx = 1  # Always long SPX
position_spx_dynamic = 1.0  # Fraction long SPX in dynamic
position_sh_dynamic = 0.0  # Fraction long SH in dynamic
position_qqq_dynamic = 1.0  # Fraction long QQQ in dynamic PSQ
position_psq_dynamic = 0.0  # Fraction long PSQ in dynamic PSQ
position_psq_alone = 0  # 0: cash, 1: long PSQ
prev_position_main = 0
prev_position_sh = 0
prev_position_vix_sh = 0
trade_count_main = 0
trade_count_sh = 0
trade_count_vix_sh = 0
trade_count_dynamic = 0  # For dynamic adjustments
trade_count_dynamic_psq = 0  # For dynamic PSQ adjustments
trade_count_psq_alone = 0
current_entry_price_main = np.nan
current_entry_price_sh = np.nan
current_entry_price_vix_sh = np.nan
current_exit_price_main = np.nan
current_exit_price_sh = np.nan
current_exit_price_vix_sh = np.nan
current_entry_price_psq_alone = np.nan
current_exit_price_psq_alone = np.nan
holding_days_main = []
holding_days_sh = []
holding_days_vix_sh = []
holding_days_psq_alone = []
entry_date_main = None
entry_date_sh = None
entry_date_vix_sh = None
entry_date_psq_alone = None
entry_price_spx_dynamic = df['Open_SPX'].iloc[0] if len(df) > 0 else np.nan
entry_price_sh_dynamic = np.nan
entry_price_qqq_dynamic = df['Open_QQQ'].iloc[0] if len(df) > 0 else np.nan
entry_price_psq_dynamic = np.nan
current_entry_date_dynamic = None
current_entry_date_dynamic_psq = None
dynamic_trades = []
dynamic_psq_trades = []
sh_trades = []
psq_alone_trades = []

# Prepare a list to collect debug data
debug_data = []

# Initialize lists for plotting
dates = []
unrealized_pnl_long_spx_values = []
sh_portfolio_values = []
long_spx_portfolio_values = []
dynamic_portfolio_values = []
dynamic_psq_portfolio_values = []
psq_alone_portfolio_values = []

# Loop over all periods to include full data in CSV, starting from 1 (to have yesterday's data)
for i in range(1, len(df)):  # Start from 1
    # Use yesterday's prices and yesterday's EMAs for conditions (data available at end of day i-1)
    spx_close_yesterday = df['Close_SPX'].iloc[i-1]
    spx_ema_yesterday = df['EMA_SPX'].iloc[i-1] if i > span else np.nan  # EMA up to day i-1
    vvix_yesterday = df['Close_VVIX'].iloc[i-1]
    vvix_ema_yesterday = df['EMA_VVIX'].iloc[i-1] if i > span else np.nan  # EMA up to day i-1
    vix_yesterday = df['Close_VIX'].iloc[i-1]
    vix_ema_yesterday = df['EMA_VIX'].iloc[i-1] if i > span else np.nan  # EMA up to day i-1
    qqq_close_yesterday = df['Close_QQQ'].iloc[i-1]
    qqq_ema_yesterday = df['EMA_QQQ'].iloc[i-1] if i > span else np.nan  # EMA up to day i-1
    sh_close_today = df['Close_SH'].iloc[i]  # Today's SH close for entry price (but signal is from yesterday)
    
    entry_condition = (vvix_yesterday > vvix_ema_yesterday and spx_close_yesterday < spx_ema_yesterday) if i > span else False
    entry_condition_vix_sh = (vix_yesterday > vix_ema_yesterday and spx_close_yesterday < spx_ema_yesterday) if i > span else False
    entry_condition_dynamic_psq = (vvix_yesterday > vvix_ema_yesterday and qqq_close_yesterday < qqq_ema_yesterday) if i > span else False
    
    # Debug print for first few iterations (adjusted for i starting at 1)
    if i < span + 6:  # Show a bit more
        print(f"i={i}, date={df.index[i]}, spx_close_yesterday={spx_close_yesterday:.2f}, spx_ema_yesterday={spx_ema_yesterday:.2f}, vvix_yesterday={vvix_yesterday:.2f}, vvix_ema_yesterday={vvix_ema_yesterday:.2f}, vix_yesterday={vix_yesterday:.2f}, vix_ema_yesterday={vix_ema_yesterday:.2f}, qqq_close_yesterday={qqq_close_yesterday:.2f}, qqq_ema_yesterday={qqq_ema_yesterday:.2f}, sh_close_today={sh_close_today:.2f}, condition={entry_condition}, condition_vix_sh={entry_condition_vix_sh}, condition_dynamic_psq={entry_condition_dynamic_psq}")
    
    # Initialize for this iteration
    exit_price_main = np.nan
    exit_price_sh = np.nan
    exit_price_vix_sh = np.nan
    exit_price_psq_alone = np.nan
    # Main strategy: Entry/Exit for short SPX (execute on day i based on yesterday's signal)
    if entry_condition and position_main == 0:
        position_main = -1
        current_entry_price_main = df['Open_SPX'].iloc[i]  # Enter at today's open
        entry_date_main = df.index[i]
        print(f"Main: Entering short SPX at {df.index[i]}")
    elif not entry_condition and position_main == -1:
        position_main = 0
        exit_price_main = df['Open_SPX'].iloc[i]  # Exit at today's open
        current_exit_price_main = exit_price_main  # For calculation
        if entry_date_main is not None:
            holding_days_main.append((df.index[i] - entry_date_main).days)
            entry_date_main = None
        print(f"Main: Exiting short SPX at {df.index[i]}")
    
    # SH strategy: Entry/Exit for long SH (execute on day i based on yesterday's signal)
    if entry_condition and position_sh == 0:
        position_sh = 1
        current_entry_price_sh = df['Open_SH'].iloc[i]  # Enter at today's open
        entry_date_sh = df.index[i]
        print(f"SH: Entering long SH at {df.index[i]}")
    elif not entry_condition and position_sh == 1:
        position_sh = 0
        exit_price_sh = df['Open_SH'].iloc[i]  # Exit at today's open
        current_exit_price_sh = exit_price_sh  # For calculation
        if entry_date_sh is not None:
            holding_days_sh.append((df.index[i] - entry_date_sh).days)
            if not np.isnan(current_exit_price_sh) and not np.isnan(current_entry_price_sh):
                trade_return_sh = ((current_exit_price_sh - current_entry_price_sh) / current_entry_price_sh) * 100
                portfolio_value_sh *= (1 + trade_return_sh / 100)  # Adjust for decimal compounding
                portfolio_returns_sh.append(trade_return_sh / 100)  # Store as decimal for stats
                sh_trades.append({
                    'Entry_Date': entry_date_sh,
                    'Exit_Date': df.index[i],
                    'Return': trade_return_sh
                })
            entry_date_sh = None
        print(f"SH: Exiting long SH at {df.index[i]}")
    # VIX-SH strategy: Entry/Exit for long SH (execute on day i based on yesterday's signal)
    if entry_condition_vix_sh and position_vix_sh == 0:
        position_vix_sh = 1
        current_entry_price_vix_sh = df['Open_SH'].iloc[i]  # Enter at today's open
        entry_date_vix_sh = df.index[i]
        print(f"VIX-SH: Entering long SH at {df.index[i]}")
    elif not entry_condition_vix_sh and position_vix_sh == 1:
        position_vix_sh = 0
        exit_price_vix_sh = df['Open_SH'].iloc[i]  # Exit at today's open
        current_exit_price_vix_sh = exit_price_vix_sh  # For calculation
        if entry_date_vix_sh is not None:
            holding_days_vix_sh.append((df.index[i] - entry_date_vix_sh).days)
            if not np.isnan(current_exit_price_vix_sh) and not np.isnan(current_entry_price_vix_sh):
                trade_return_vix_sh = (current_exit_price_vix_sh - current_entry_price_vix_sh) / current_entry_price_vix_sh
                portfolio_value_vix_sh *= (1 + trade_return_vix_sh)
                portfolio_returns_vix_sh.append(trade_return_vix_sh)
            entry_date_vix_sh = None
        print(f"VIX-SH: Exiting long SH at {df.index[i]}")

    # PSQ Alone strategy: Entry/Exit for long PSQ (execute on day i based on yesterday's signal)
    if entry_condition_dynamic_psq and position_psq_alone == 0:
        position_psq_alone = 1
        current_entry_price_psq_alone = df['Open_PSQ'].iloc[i]  # Enter at today's open
        entry_date_psq_alone = df.index[i]
        print(f"PSQ Alone: Entering long PSQ at {df.index[i]}")
    elif not entry_condition_dynamic_psq and position_psq_alone == 1:
        position_psq_alone = 0
        exit_price_psq_alone = df['Open_PSQ'].iloc[i]  # Exit at today's open
        current_exit_price_psq_alone = exit_price_psq_alone  # For calculation
        if entry_date_psq_alone is not None:
            holding_days_psq_alone.append((df.index[i] - entry_date_psq_alone).days)
            if not np.isnan(current_exit_price_psq_alone) and not np.isnan(current_entry_price_psq_alone):
                trade_return_psq_alone = ((current_exit_price_psq_alone - current_entry_price_psq_alone) / current_entry_price_psq_alone) * 100
                portfolio_value_psq_alone *= (1 + trade_return_psq_alone / 100)  # Adjust for decimal compounding
                portfolio_returns_psq_alone.append(trade_return_psq_alone / 100)  # Store as decimal for stats
                psq_alone_trades.append({
                    'Entry_Date': entry_date_psq_alone,
                    'Exit_Date': df.index[i],
                    'Return': trade_return_psq_alone
                })
            entry_date_psq_alone = None
        print(f"PSQ Alone: Exiting long PSQ at {df.index[i]}")

    # Count trades
    if position_main == -1 and prev_position_main == 0:
        trade_count_main += 1
    if position_sh == 1 and prev_position_sh == 0:
        trade_count_sh += 1
    if position_vix_sh == 1 and prev_position_vix_sh == 0:
        trade_count_vix_sh += 1
    if position_psq_alone == 1 and prev_position_psq_alone == 0:
        trade_count_psq_alone += 1
    
    prev_position_main = position_main
    prev_position_sh = position_sh
    prev_position_vix_sh = position_vix_sh
    prev_position_psq_alone = position_psq_alone
    
    # Calculate trade returns on exit (no daily returns; hold until exit)
    trade_return_main = 0
    if not np.isnan(current_exit_price_main) and not np.isnan(current_entry_price_main):
        trade_return_main = - (current_exit_price_main - current_entry_price_main) / current_entry_price_main
        portfolio_value_main *= (1 + trade_return_main)
        portfolio_returns_main.append(trade_return_main)
        current_entry_price_main = np.nan
        current_exit_price_main = np.nan
    
    # trade_return_sh and trade_return_psq_alone are now calculated inside their respective elif blocks
    
    # For days in position but not exiting, no return applied (cash otherwise)
    daily_portfolio_return_main = trade_return_main  # Only on exit day
    daily_portfolio_return_sh = trade_return_sh if 'trade_return_sh' in locals() else 0  # Only on exit day
    daily_portfolio_return_vix_sh = trade_return_vix_sh if 'trade_return_vix_sh' in locals() else 0  # Only on exit day
    daily_portfolio_return_psq_alone = trade_return_psq_alone if 'trade_return_psq_alone' in locals() else 0  # Only on exit day
    
    # Long Only SPX strategy: Always long SPX, apply daily returns
    daily_return_long_spx = df['Return_SPX'].iloc[i]
    portfolio_value_long_spx *= (1 + daily_return_long_spx)
    portfolio_returns_long_spx.append(daily_return_long_spx)
    
    # Dynamic SH Strategy: Start with 100% SPX, adjust on SH signals
    if entry_condition and position_sh_dynamic == 0.0:
        # Enter: Reduce SPX to 80%, add 20% SH
        spx_value = position_spx_dynamic * portfolio_value_dynamic
        sh_value = position_sh_dynamic * portfolio_value_dynamic
        new_spx_value = spx_value * 0.8
        new_sh_value = sh_value + (spx_value * 0.2)
        total_value = new_spx_value + new_sh_value
        position_spx_dynamic = new_spx_value / total_value
        position_sh_dynamic = new_sh_value / total_value
        entry_price_sh_dynamic = df['Open_SH'].iloc[i]
        current_entry_date_dynamic = df.index[i]
        trade_count_dynamic += 1
        print(f"Dynamic: Adjusting to 80% SPX + 20% SH at {df.index[i]}")
    elif not entry_condition and position_sh_dynamic > 0.0:
        # Exit: Close SH, back to 100% SPX
        spx_value = position_spx_dynamic * portfolio_value_dynamic
        sh_value = position_sh_dynamic * portfolio_value_dynamic
        new_spx_value = spx_value + sh_value
        total_value = new_spx_value
        position_spx_dynamic = 1.0
        position_sh_dynamic = 0.0
        entry_price_sh_dynamic = np.nan
        if current_entry_date_dynamic is not None:
            start_idx = dates.index(current_entry_date_dynamic)
            end_idx = len(dates)  # Current i is len(dates) since dates.append after
            cumulative_return = np.prod(1 + np.array(portfolio_returns_dynamic[start_idx:end_idx])) - 1
            dynamic_trades.append({
                'Entry_Date': current_entry_date_dynamic,
                'Exit_Date': df.index[i],
                'Return': cumulative_return
            })
            current_entry_date_dynamic = None
        trade_count_dynamic += 1
        print(f"Dynamic: Back to 100% SPX at {df.index[i]}")
    
    # Dynamic PSQ Strategy: Start with 100% QQQ, adjust on PSQ signals
    if entry_condition_dynamic_psq and position_psq_dynamic == 0.0:
        # Enter: Reduce QQQ to 80%, add 20% PSQ
        qqq_value = position_qqq_dynamic * portfolio_value_dynamic_psq
        psq_value = position_psq_dynamic * portfolio_value_dynamic_psq
        new_qqq_value = qqq_value * 0.8
        new_psq_value = psq_value + (qqq_value * 0.2)
        total_value = new_qqq_value + new_psq_value
        position_qqq_dynamic = new_qqq_value / total_value
        position_psq_dynamic = new_psq_value / total_value
        entry_price_psq_dynamic = df['Open_PSQ'].iloc[i]
        current_entry_date_dynamic_psq = df.index[i]
        trade_count_dynamic_psq += 1
        print(f"Dynamic PSQ: Adjusting to 80% QQQ + 20% PSQ at {df.index[i]}")
    elif not entry_condition_dynamic_psq and position_psq_dynamic > 0.0:
        # Exit: Close PSQ, back to 100% QQQ
        qqq_value = position_qqq_dynamic * portfolio_value_dynamic_psq
        psq_value = position_psq_dynamic * portfolio_value_dynamic_psq
        new_qqq_value = qqq_value + psq_value
        total_value = new_qqq_value
        position_qqq_dynamic = 1.0
        position_psq_dynamic = 0.0
        entry_price_psq_dynamic = np.nan
        if current_entry_date_dynamic_psq is not None:
            start_idx = dates.index(current_entry_date_dynamic_psq)
            end_idx = len(dates)  # Current i is len(dates) since dates.append after
            cumulative_return_psq = np.prod(1 + np.array(portfolio_returns_dynamic_psq[start_idx:end_idx])) - 1
            dynamic_psq_trades.append({
                'Entry_Date': current_entry_date_dynamic_psq,
                'Exit_Date': df.index[i],
                'Return': cumulative_return_psq
            })
            current_entry_date_dynamic_psq = None
        trade_count_dynamic_psq += 1
        print(f"Dynamic PSQ: Back to 100% QQQ at {df.index[i]}")
    
    # Apply daily returns for dynamic strategy
    daily_return_dynamic = position_spx_dynamic * df['Return_SPX'].iloc[i] + position_sh_dynamic * df['Return_SH'].iloc[i]
    portfolio_value_dynamic *= (1 + daily_return_dynamic)
    portfolio_returns_dynamic.append(daily_return_dynamic)
    
    # Apply daily returns for dynamic PSQ strategy
    daily_return_dynamic_psq = position_qqq_dynamic * df['Return_QQQ'].iloc[i] + position_psq_dynamic * df['Return_PSQ'].iloc[i]
    portfolio_value_dynamic_psq *= (1 + daily_return_dynamic_psq)
    portfolio_returns_dynamic_psq.append(daily_return_dynamic_psq)
    
    # Calculate unrealized trade returns (for open positions)
    unrealized_trade_return_main = - (df['Close_SPX'].iloc[i] - current_entry_price_main) / current_entry_price_main if not np.isnan(current_entry_price_main) else 0
    unrealized_trade_return_sh = (df['Close_SH'].iloc[i] - current_entry_price_sh) / current_entry_price_sh if not np.isnan(current_entry_price_sh) else 0
    unrealized_trade_return_vix_sh = (df['Close_SH'].iloc[i] - current_entry_price_vix_sh) / current_entry_price_vix_sh if not np.isnan(current_entry_price_vix_sh) else 0
    unrealized_trade_return_psq = (df['Close_PSQ'].iloc[i] - entry_price_psq_dynamic) / entry_price_psq_dynamic if not np.isnan(entry_price_psq_dynamic) else 0
    unrealized_trade_return_psq_alone = (df['Close_PSQ'].iloc[i] - current_entry_price_psq_alone) / current_entry_price_psq_alone if not np.isnan(current_entry_price_psq_alone) else 0
    
    # Calculate unrealized PnL for each strategy
    unrealized_pnl_main = unrealized_trade_return_main * portfolio_value_main if position_main != 0 else 0
    unrealized_pnl_sh = unrealized_trade_return_sh * portfolio_value_sh if position_sh == 1 else 0
    unrealized_pnl_vix_sh = unrealized_trade_return_vix_sh * portfolio_value_vix_sh if position_vix_sh == 1 else 0
    unrealized_pnl_long_spx = portfolio_value_long_spx - initial_capital
    unrealized_pnl_dynamic = (position_spx_dynamic * (df['Close_SPX'].iloc[i] - entry_price_spx_dynamic) / entry_price_spx_dynamic * portfolio_value_dynamic + position_sh_dynamic * (df['Close_SH'].iloc[i] - entry_price_sh_dynamic) / entry_price_sh_dynamic * portfolio_value_dynamic) if not np.isnan(entry_price_sh_dynamic) else position_spx_dynamic * (df['Close_SPX'].iloc[i] - entry_price_spx_dynamic) / entry_price_spx_dynamic * portfolio_value_dynamic
    unrealized_pnl_dynamic_psq = (position_qqq_dynamic * (df['Close_QQQ'].iloc[i] - entry_price_qqq_dynamic) / entry_price_qqq_dynamic * portfolio_value_dynamic_psq + position_psq_dynamic * (df['Close_PSQ'].iloc[i] - entry_price_psq_dynamic) / entry_price_psq_dynamic * portfolio_value_dynamic_psq) if not np.isnan(entry_price_psq_dynamic) else position_qqq_dynamic * (df['Close_QQQ'].iloc[i] - entry_price_qqq_dynamic) / entry_price_qqq_dynamic * portfolio_value_dynamic_psq
    unrealized_pnl_psq_alone = unrealized_trade_return_psq_alone * portfolio_value_psq_alone if position_psq_alone == 1 else 0
    
    # Collect data for plotting
    dates.append(df.index[i])
    unrealized_pnl_long_spx_values.append(unrealized_pnl_long_spx)
    sh_portfolio_values.append(portfolio_value_sh)
    long_spx_portfolio_values.append(portfolio_value_long_spx)
    dynamic_portfolio_values.append(portfolio_value_dynamic)
    dynamic_psq_portfolio_values.append(portfolio_value_dynamic_psq)
    psq_alone_portfolio_values.append(portfolio_value_psq_alone)

    # Collect debug data with all indicators, signals, and trade details for both strategies
    debug_data.append({
        'Date': df.index[i],
        'SPX_Open': df['Open_SPX'].iloc[i],
        'SPX_Close': df['Close_SPX'].iloc[i],
        'SPX_Return': df['Return_SPX'].iloc[i],
        'EMA_SPX': df['EMA_SPX'].iloc[i],
        'SPX_EMA_Yesterday': spx_ema_yesterday,
        'VVIX_Open': df['Open_VVIX'].iloc[i],
        'VVIX': df['Close_VVIX'].iloc[i],
        'EMA_VVIX': df['EMA_VVIX'].iloc[i],
        'VVIX_EMA_Yesterday': vvix_ema_yesterday,
        'VIX_Open': df['Open_VIX'].iloc[i],
        'VIX': df['Close_VIX'].iloc[i],
        'EMA_VIX': df['EMA_VIX'].iloc[i],
        'VIX_EMA_Yesterday': vix_ema_yesterday,
        'QQQ_Open': df['Open_QQQ'].iloc[i],
        'QQQ_Close': df['Close_QQQ'].iloc[i],
        'QQQ_Return': df['Return_QQQ'].iloc[i],
        'EMA_QQQ': df['EMA_QQQ'].iloc[i],
        'QQQ_EMA_Yesterday': qqq_ema_yesterday,
        'SH_Open': df['Open_SH'].iloc[i],
        'SH_Close': sh_close_today,
        'SH_Return': df['Return_SH'].iloc[i],
        'PSQ_Open': df['Open_PSQ'].iloc[i],
        'PSQ_Close': df['Close_PSQ'].iloc[i],
        'PSQ_Return': df['Return_PSQ'].iloc[i],
        'Entry_Condition': entry_condition,
        'Entry_Condition_VIX_SH': entry_condition_vix_sh,
        'Entry_Condition_Dynamic_PSQ': entry_condition_dynamic_psq,
        'Position_Main': position_main,
        'Trade_Entered_Main': 1 if position_main == -1 and prev_position_main == 0 else 0,
        'Trade_Exited_Main': 1 if position_main == 0 and prev_position_main == -1 else 0,
        'Entry_Price_Main': current_entry_price_main,
        'Exit_Price_Main': exit_price_main,
        'Trade_Return_Main': trade_return_main,
        'Unrealized_Trade_Return_Main': unrealized_trade_return_main,
        'Daily_Portfolio_Return_Main': daily_portfolio_return_main,
        'Portfolio_Value_Main': portfolio_value_main,
        'Position_SH': position_sh,
        'Trade_Entered_SH': 1 if position_sh == 1 and prev_position_sh == 0 else 0,
        'Trade_Exited_SH': 1 if position_sh == 0 and prev_position_sh == 1 else 0,
        'Entry_Price_SH': current_entry_price_sh,
        'Exit_Price_SH': exit_price_sh,
        'Trade_Return_SH': trade_return_sh if 'trade_return_sh' in locals() else 0,
        'Unrealized_Trade_Return_SH': unrealized_trade_return_sh,
        'Daily_Portfolio_Return_SH': daily_portfolio_return_sh,
        'Portfolio_Value_SH': portfolio_value_sh,
        'Position_VIX_SH': position_vix_sh,
        'Trade_Entered_VIX_SH': 1 if position_vix_sh == 1 and prev_position_vix_sh == 0 else 0,
        'Trade_Exited_VIX_SH': 1 if position_vix_sh == 0 and prev_position_vix_sh == 1 else 0,
        'Entry_Price_VIX_SH': current_entry_price_vix_sh,
        'Exit_Price_VIX_SH': exit_price_vix_sh,
        'Trade_Return_VIX_SH': trade_return_vix_sh if 'trade_return_vix_sh' in locals() else 0,
        'Unrealized_Trade_Return_VIX_SH': unrealized_trade_return_vix_sh,
        'Daily_Portfolio_Return_VIX_SH': daily_portfolio_return_vix_sh,
        'Portfolio_Value_VIX_SH': portfolio_value_vix_sh,
        'Position_Long_SPX': position_long_spx,
        'Portfolio_Value_Long_SPX': portfolio_value_long_spx,
        'Position_SPX_Dynamic': position_spx_dynamic,
        'Position_SH_Dynamic': position_sh_dynamic,
        'Portfolio_Value_Dynamic': portfolio_value_dynamic,
        'Position_QQQ_Dynamic': position_qqq_dynamic,
        'Position_PSQ_Dynamic': position_psq_dynamic,
        'Portfolio_Value_Dynamic_PSQ': portfolio_value_dynamic_psq,
        'Entry_Price_PSQ_Dynamic': entry_price_psq_dynamic,
        'Exit_Price_PSQ_Dynamic': np.nan,  # No specific exit price for dynamic strategy
        'Unrealized_PnL_Dynamic_PSQ': unrealized_pnl_dynamic_psq,
        'Position_PSQ_Alone': position_psq_alone,
        'Trade_Entered_PSQ_Alone': 1 if position_psq_alone == 1 and prev_position_psq_alone == 0 else 0,
        'Trade_Exited_PSQ_Alone': 1 if position_psq_alone == 0 and prev_position_psq_alone == 1 else 0,
        'Entry_Price_PSQ_Alone': current_entry_price_psq_alone,
        'Exit_Price_PSQ_Alone': exit_price_psq_alone,
        'Trade_Return_PSQ_Alone': trade_return_psq_alone if 'trade_return_psq_alone' in locals() else 0,
        'Unrealized_Trade_Return_PSQ_Alone': unrealized_trade_return_psq_alone,
        'Daily_Portfolio_Return_PSQ_Alone': daily_portfolio_return_psq_alone,
        'Portfolio_Value_PSQ_Alone': portfolio_value_psq_alone,
        'Unrealized_PnL_PSQ_Alone': unrealized_pnl_psq_alone,
    })

# Create DataFrame and save to CSV with full data and backtest details
debug_df = pd.DataFrame(debug_data)
debug_df.to_csv('full_data_with_indicators.csv', index=False)

# Debug: Print trade counts
print(f"Number of trades (Main): {trade_count_main}")
print(f"Number of trades (SH): {trade_count_sh}")
print(f"Number of trades (VIX-SH): {trade_count_vix_sh}")
print(f"Number of trades (PSQ Alone): {trade_count_psq_alone}")

# Function to calculate stats
def calculate_stats(portfolio_returns, portfolio_value, initial_capital, start_date, end_date):
    total_return = (portfolio_value - initial_capital) / initial_capital
    total_days = (end_date - start_date).days
    years = total_days / 365.25
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0
    if len(portfolio_returns) > 0:
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Approximate annualization assuming trade frequency
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

        # Max drawdown based on cumulative trade returns
        cumulative = np.cumprod(1 + np.array(portfolio_returns))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
    else:
        volatility = 0
        sharpe_ratio = 0
        max_drawdown = 0
    return total_return, annualized_return, volatility, sharpe_ratio, max_drawdown

# Calculate and output stats for both strategies
start_date = df.index[0]
end_date = df.index[-1]
total_days = (end_date - start_date).days
years = total_days / 365.25
average_trades_per_year_main = trade_count_main / years if years > 0 else 0
average_trades_per_year_sh = trade_count_sh / years if years > 0 else 0
average_trades_per_year_vix_sh = trade_count_vix_sh / years if years > 0 else 0
average_trades_per_year_long_spx = trade_count_dynamic / years if years > 0 else 0
average_trades_per_year_dynamic_psq = trade_count_dynamic_psq / years if years > 0 else 0
average_trades_per_year_psq_alone = trade_count_psq_alone / years if years > 0 else 0
average_holding_days_main = np.mean(holding_days_main) if holding_days_main else 0
average_holding_days_sh = np.mean(holding_days_sh) if holding_days_sh else 0
average_holding_days_vix_sh = np.mean(holding_days_vix_sh) if holding_days_vix_sh else 0
average_holding_days_psq_alone = np.mean(holding_days_psq_alone) if holding_days_psq_alone else 0
total_return_main, annualized_return_main, volatility_main, sharpe_ratio_main, max_drawdown_main = calculate_stats(portfolio_returns_main, portfolio_value_main, initial_capital, start_date, end_date)
total_return_sh, annualized_return_sh, volatility_sh, sharpe_ratio_sh, max_drawdown_sh = calculate_stats(portfolio_returns_sh, portfolio_value_sh, initial_capital, start_date, end_date)
total_return_vix_sh, annualized_return_vix_sh, volatility_vix_sh, sharpe_ratio_vix_sh, max_drawdown_vix_sh = calculate_stats(portfolio_returns_vix_sh, portfolio_value_vix_sh, initial_capital, start_date, end_date)
total_return_long_spx, annualized_return_long_spx, volatility_long_spx, sharpe_ratio_long_spx, max_drawdown_long_spx = calculate_stats(portfolio_returns_long_spx, portfolio_value_long_spx, initial_capital, start_date, end_date)
total_return_dynamic, annualized_return_dynamic, volatility_dynamic, sharpe_ratio_dynamic, max_drawdown_dynamic = calculate_stats(portfolio_returns_dynamic, portfolio_value_dynamic, initial_capital, start_date, end_date)
total_return_dynamic_psq, annualized_return_dynamic_psq, volatility_dynamic_psq, sharpe_ratio_dynamic_psq, max_drawdown_dynamic_psq = calculate_stats(portfolio_returns_dynamic_psq, portfolio_value_dynamic_psq, initial_capital, start_date, end_date)
total_return_psq_alone, annualized_return_psq_alone, volatility_psq_alone, sharpe_ratio_psq_alone, max_drawdown_psq_alone = calculate_stats(portfolio_returns_psq_alone, portfolio_value_psq_alone, initial_capital, start_date, end_date)

print("\nMain Strategy (Short SPX):")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value: ${portfolio_value_main:.2f}")
print(f"Total Return: {total_return_main:.2%}")
print(f"Annualized Return: {annualized_return_main:.2%}")
print(f"Volatility (Annualized): {volatility_main:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_main:.2f}")
print(f"Max Drawdown: {max_drawdown_main:.2%}")
print(f"Average Trades per Year: {average_trades_per_year_main:.2f}")
print(f"Average Holding Days per Trade: {average_holding_days_main:.2f}")

print("\nSH Strategy (Long SH):")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value: ${portfolio_value_sh:.2f}")
print(f"Total Return: {total_return_sh:.2%}")
print(f"Annualized Return: {annualized_return_sh:.2%}")
print(f"Volatility (Annualized): {volatility_sh:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_sh:.2f}")
print(f"Max Drawdown: {max_drawdown_sh:.2%}")
print(f"Average Trades per Year: {average_trades_per_year_sh:.2f}")
print(f"Average Holding Days per Trade: {average_holding_days_sh:.2f}")

print("\nVIX-SH Strategy (Long SH):")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value: ${portfolio_value_vix_sh:.2f}")
print(f"Total Return: {total_return_vix_sh:.2%}")
print(f"Annualized Return: {annualized_return_vix_sh:.2%}")
print(f"Volatility (Annualized): {volatility_vix_sh:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_vix_sh:.2f}")
print(f"Max Drawdown: {max_drawdown_vix_sh:.2%}")
print(f"Average Trades per Year: {average_trades_per_year_vix_sh:.2f}")
print(f"Average Holding Days per Trade: {average_holding_days_vix_sh:.2f}")

print("\nLong Only SPX Strategy:")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value: ${portfolio_value_long_spx:.2f}")
print(f"Total Return: {total_return_long_spx:.2%}")
print(f"Annualized Return: {annualized_return_long_spx:.2%}")
print(f"Volatility (Annualized): {volatility_long_spx:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_long_spx:.2f}")
print(f"Max Drawdown: {max_drawdown_long_spx:.2%}")

print("\nDynamic SH Strategy (Long SH):")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value: ${portfolio_value_dynamic:.2f}")
print(f"Total Return: {total_return_dynamic:.2%}")
print(f"Annualized Return: {annualized_return_dynamic:.2%}")
print(f"Volatility (Annualized): {volatility_dynamic:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_dynamic:.2f}")
print(f"Max Drawdown: {max_drawdown_dynamic:.2%}")
print(f"Number of Adjustments: {trade_count_dynamic}")

print("\nDynamic PSQ Strategy (Long PSQ):")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value: ${portfolio_value_dynamic_psq:.2f}")
print(f"Total Return: {total_return_dynamic_psq:.2%}")
print(f"Annualized Return: {annualized_return_dynamic_psq:.2%}")
print(f"Volatility (Annualized): {volatility_dynamic_psq:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_dynamic_psq:.2f}")
print(f"Max Drawdown: {max_drawdown_dynamic_psq:.2%}")
print(f"Number of Adjustments: {trade_count_dynamic_psq}")

print("\nPSQ Alone Strategy (Long PSQ):")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value: ${portfolio_value_psq_alone:.2f}")
print(f"Total Return: {total_return_psq_alone:.2%}")
print(f"Annualized Return: {annualized_return_psq_alone:.2%}")
print(f"Volatility (Annualized): {volatility_psq_alone:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_psq_alone:.2f}")
print(f"Max Drawdown: {max_drawdown_psq_alone:.2%}")
print(f"Average Trades per Year: {average_trades_per_year_psq_alone:.2f}")
print(f"Average Holding Days per Trade: {average_holding_days_psq_alone:.2f}")

# Plot the Unrealized PnL for Long Only SPX and SH Strategy Portfolio Value
plt.figure(figsize=(10, 6))
plt.plot(dates, unrealized_pnl_long_spx_values, label='Unrealized PnL Long Only SPX')
plt.plot(dates, sh_portfolio_values, label='SH Strategy Portfolio Value')
plt.plot(dates, psq_alone_portfolio_values, label='PSQ Alone Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Unrealized PnL Long Only SPX, SH Strategy, and PSQ Alone Portfolio Value Over Time')
plt.legend()
plt.show()

# Plot the Daily Portfolio Value for Long Only SPX and Dynamic SH strategies
plt.figure(figsize=(10, 6))
plt.plot(dates, long_spx_portfolio_values, label='Portfolio Value Long Only SPX')
plt.plot(dates, dynamic_portfolio_values, label='Portfolio Value Dynamic SH')
plt.plot(dates, dynamic_psq_portfolio_values, label='Portfolio Value Dynamic PSQ')
plt.plot(dates, psq_alone_portfolio_values, label='Portfolio Value PSQ Alone')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Daily Portfolio Value Over Time: Long Only SPX, Dynamic SH, Dynamic PSQ, and PSQ Alone')
plt.legend()
plt.show()

# Plot the Daily Returns for Long Only SPX and Dynamic SH strategies
plt.figure(figsize=(10, 6))
plt.plot(dates, portfolio_returns_long_spx, label='Daily Return Long Only SPX')
plt.plot(dates, portfolio_returns_dynamic, label='Daily Return Dynamic SH')
plt.plot(dates, portfolio_returns_dynamic_psq, label='Daily Return Dynamic PSQ')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('Daily Returns Over Time: Long Only SPX, Dynamic SH, and Dynamic PSQ')
plt.legend()
plt.show()

# After the loop, create trade data for CSV
sh_trade_df = pd.DataFrame(sh_trades)
sh_trade_df.to_csv('sh_alone_trades.csv', index=False)

psq_alone_trade_df = pd.DataFrame(psq_alone_trades)
psq_alone_trade_df.to_csv('psq_alone_trades.csv', index=False)