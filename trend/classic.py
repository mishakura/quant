import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import os
from datetime import datetime

# Download AAPL data
def get_stock_data(ticker="AAPL", period="5y", interval="1d"):
    """Download stock data using yfinance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

# Calculate technical indicators
def add_indicators(df, atr_length=14, dc_length_1=200, dc_length_2=100):
    """Add ATR and Donchian Channels to dataframe"""
    # Calculate ATR
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=atr_length)
    
    # Calculate Donchian Channels (200 period)
    dc_200 = ta.donchian(df["High"], df["Low"], lower_length=dc_length_1, upper_length=dc_length_1)
    df["dc_upper_200"] = dc_200[f"DCU_{dc_length_1}_{dc_length_1}"]
    df["dc_lower_200"] = dc_200[f"DCL_{dc_length_1}_{dc_length_1}"]
    
    # Calculate Donchian Channels (100 period)
    dc_100 = ta.donchian(df["High"], df["Low"], lower_length=dc_length_2, upper_length=dc_length_2)
    df["dc_upper_100"] = dc_100[f"DCU_{dc_length_2}_{dc_length_2}"]
    df["dc_lower_100"] = dc_100[f"DCL_{dc_length_2}_{dc_length_2}"]
    
    return df

# Calculate trading signals with position sizing
def add_trading_signals(df, initial_capital=100000, risk_pct=0.01):
    """
    Add trading signals based on Donchian channel breakouts and ATR stop losses:
    - Signal generated when Close crosses Donchian bands
    - ATR-based stop loss calculated at entry (ATR * 3)
    - Position size based on risk percentage of capital
    - Actual entry/exit occurs at the NEXT day's OPEN price
    """
    # Initialize columns for signal generation
    df['long_entry_signal'] = False
    df['long_exit_signal'] = False
    df['short_entry_signal'] = False
    df['short_exit_signal'] = False
    
    # Initialize columns for actual positions and execution
    df['signal'] = 0  # 0=no position, 1=long, -1=short
    df['position_status'] = 'NO POSITION'
    df['entry_price'] = None
    df['exit_price'] = None
    df['entry_atr'] = None
    df['stop_loss'] = None
    df['exit_reason'] = None
    
    # Portfolio tracking columns
    df['shares'] = 0
    df['position_value'] = 0
    df['trade_pnl'] = 0
    df['capital'] = initial_capital
    df['equity'] = initial_capital
    
    # Trade statistics
    trades = []
    current_trade = None
    
    # Skip the first row since we need previous values
    for i in range(1, len(df)):
        prev_idx = i-1
        # Defensive: skip signals for first 200 rows (Donchian bands not valid)
        if prev_idx < 200:
            continue
        
        # Check conditions against previous day's Donchian values (signal generation)
        # Long conditions
        long_entry_signal = df['Close'].iloc[i] > df['dc_upper_200'].iloc[prev_idx]
        long_exit_signal = df['Close'].iloc[i] < df['dc_lower_100'].iloc[prev_idx]
        
        # Short conditions
        short_entry_signal = df['Close'].iloc[i] < df['dc_lower_200'].iloc[prev_idx]
        short_exit_signal = df['Close'].iloc[i] > df['dc_upper_100'].iloc[prev_idx]
        
        # Store signal conditions
        df.loc[df.index[i], 'long_entry_signal'] = long_entry_signal
        df.loc[df.index[i], 'long_exit_signal'] = long_exit_signal
        df.loc[df.index[i], 'short_entry_signal'] = short_entry_signal
        df.loc[df.index[i], 'short_exit_signal'] = short_exit_signal
        
        # Previous signal state
        prev_signal = df['signal'].iloc[prev_idx]
        prev_capital = df['capital'].iloc[prev_idx]
        prev_shares = df['shares'].iloc[prev_idx]
        
        # Copy previous capital by default
        df.loc[df.index[i], 'capital'] = prev_capital
        
        # Trading logic for today, based on yesterday's signals
        # (we need enough history)
        if i > 1:
            # Get previous signals
            prev_long_entry = df['long_entry_signal'].iloc[prev_idx]
            prev_long_exit = df['long_exit_signal'].iloc[prev_idx]
            prev_short_entry = df['short_entry_signal'].iloc[prev_idx]
            prev_short_exit = df['short_exit_signal'].iloc[prev_idx]
            
            # Previous position
            prev_position = df['signal'].iloc[prev_idx]
            
            # LONG ENTRY: Yesterday's close gave entry signal while not in position
            if prev_long_entry and prev_position == 0:
                entry_price = df['Open'].iloc[i]
                entry_atr = df['atr'].iloc[prev_idx]  # ATR at signal generation
                stop_loss = entry_price - (entry_atr * 3)
                
                # Calculate position size based on risk (use ATR only, not ATR*3)
                risk_amount = prev_capital * risk_pct
                risk_per_share = entry_atr  # ATR only
                shares = int(risk_amount / risk_per_share)
                position_value = shares * entry_price

                # Ensure we don't use more than available capital
                if position_value > prev_capital:
                    shares = int(prev_capital / entry_price)
                    position_value = shares * entry_price
                
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'position_status'] = 'LONG (New Entry)'
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'entry_atr'] = entry_atr
                df.loc[df.index[i], 'stop_loss'] = stop_loss
                df.loc[df.index[i], 'shares'] = shares
                df.loc[df.index[i], 'position_value'] = position_value
                df.loc[df.index[i], 'equity'] = prev_capital
                
                # Record trade start
                current_trade = {
                    'entry_date': df.index[i],
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'entry_atr': entry_atr
                }
            
            # LONG EXIT: Yesterday's close gave exit signal while in long position
            elif (prev_long_exit and prev_position == 1) or \
                 (prev_position == 1 and df['Close'].iloc[prev_idx] < df['stop_loss'].iloc[prev_idx]):
                
                # Determine exit reason
                if prev_long_exit:
                    exit_reason = "Donchian Exit"
                else:
                    exit_reason = "Stop Loss"
                
                exit_price = df['Open'].iloc[i]
                
                # Calculate P&L
                prev_entry_price = df['entry_price'].iloc[prev_idx]
                prev_shares = df['shares'].iloc[prev_idx]
                trade_pnl = prev_shares * (exit_price - prev_entry_price)
                new_capital = prev_capital + trade_pnl
                
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'position_status'] = f'NO POSITION (Long Exit - {exit_reason})'
                df.loc[df.index[i], 'exit_price'] = exit_price
                df.loc[df.index[i], 'exit_reason'] = exit_reason
                df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                df.loc[df.index[i], 'shares'] = 0
                df.loc[df.index[i], 'position_value'] = 0
                df.loc[df.index[i], 'capital'] = new_capital
                df.loc[df.index[i], 'equity'] = new_capital
                
                # Complete the trade record
                if current_trade:
                    current_trade['exit_date'] = df.index[i]
                    current_trade['exit_price'] = exit_price
                    current_trade['exit_reason'] = exit_reason
                    current_trade['pnl'] = trade_pnl
                    current_trade['return_pct'] = (trade_pnl / (current_trade['entry_price'] * current_trade['shares'])) * 100
                    trades.append(current_trade)
                    current_trade = None
            
            # SHORT ENTRY: Yesterday's close gave short entry signal while not in position
            elif prev_short_entry and prev_position == 0:
                entry_price = df['Open'].iloc[i]
                entry_atr = df['atr'].iloc[prev_idx]  # ATR at signal generation
                stop_loss = entry_price + (entry_atr * 3)
                
                # Calculate position size based on risk (use ATR only, not ATR*3)
                risk_amount = prev_capital * risk_pct
                risk_per_share = entry_atr  # ATR only
                shares = int(risk_amount / risk_per_share)
                position_value = shares * entry_price

                # Ensure we don't use more than available capital
                if position_value > prev_capital:
                    shares = int(prev_capital / entry_price)
                    position_value = shares * entry_price
                
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'position_status'] = 'SHORT (New Entry)'
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'entry_atr'] = entry_atr
                df.loc[df.index[i], 'stop_loss'] = stop_loss
                df.loc[df.index[i], 'shares'] = shares
                df.loc[df.index[i], 'position_value'] = position_value
                df.loc[df.index[i], 'equity'] = prev_capital
                
                # Record trade start
                current_trade = {
                    'entry_date': df.index[i],
                    'direction': 'SHORT',
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'entry_atr': entry_atr
                }
            
            # SHORT EXIT: Yesterday's close gave short exit signal while in short position
            elif (prev_short_exit and prev_position == -1) or \
                 (prev_position == -1 and df['Close'].iloc[prev_idx] > df['stop_loss'].iloc[prev_idx]):
                
                # Determine exit reason
                if prev_short_exit:
                    exit_reason = "Donchian Exit"
                else:
                    exit_reason = "Stop Loss"
                
                exit_price = df['Open'].iloc[i]
                
                # Calculate P&L for short position
                prev_entry_price = df['entry_price'].iloc[prev_idx]
                prev_shares = df['shares'].iloc[prev_idx]
                trade_pnl = prev_shares * (prev_entry_price - exit_price)
                new_capital = prev_capital + trade_pnl
                
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'position_status'] = f'NO POSITION (Short Exit - {exit_reason})'
                df.loc[df.index[i], 'exit_price'] = exit_price
                df.loc[df.index[i], 'exit_reason'] = exit_reason
                df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                df.loc[df.index[i], 'shares'] = 0
                df.loc[df.index[i], 'position_value'] = 0
                df.loc[df.index[i], 'capital'] = new_capital
                df.loc[df.index[i], 'equity'] = new_capital
                
                # Complete the trade record
                if current_trade:
                    current_trade['exit_date'] = df.index[i]
                    current_trade['exit_price'] = exit_price
                    current_trade['exit_reason'] = exit_reason
                    current_trade['pnl'] = trade_pnl
                    current_trade['return_pct'] = (trade_pnl / (current_trade['entry_price'] * current_trade['shares'])) * 100
                    trades.append(current_trade)
                    current_trade = None
            
            # Otherwise maintain previous position and update equity value
            else:
                df.loc[df.index[i], 'signal'] = prev_position
                df.loc[df.index[i], 'stop_loss'] = df['stop_loss'].iloc[prev_idx]
                df.loc[df.index[i], 'entry_price'] = df['entry_price'].iloc[prev_idx]
                df.loc[df.index[i], 'entry_atr'] = df['entry_atr'].iloc[prev_idx]
                df.loc[df.index[i], 'shares'] = prev_shares
                
                if prev_position == 1:
                    df.loc[df.index[i], 'position_status'] = 'LONG'
                    # Update position value and unrealized P&L
                    current_price = df['Close'].iloc[i]
                    position_value = prev_shares * current_price
                    df.loc[df.index[i], 'position_value'] = position_value
                    df.loc[df.index[i], 'equity'] = prev_capital + (position_value - (prev_shares * df['entry_price'].iloc[prev_idx]))
                
                elif prev_position == -1:
                    df.loc[df.index[i], 'position_status'] = 'SHORT'
                    # Update position value and unrealized P&L
                    current_price = df['Close'].iloc[i]
                    position_value = prev_shares * current_price
                    df.loc[df.index[i], 'position_value'] = position_value
                    df.loc[df.index[i], 'equity'] = prev_capital + (prev_shares * (df['entry_price'].iloc[prev_idx] - current_price))
                
                else:
                    df.loc[df.index[i], 'position_status'] = 'NO POSITION'
                    df.loc[df.index[i], 'equity'] = prev_capital
    
    # Calculate performance metrics
    performance = calculate_performance(df, trades)
    
    return df, trades, performance

# Calculate performance metrics
def calculate_performance(df, trades):
    """Calculate key performance metrics for the trading strategy"""
    performance = {}
    
    if not trades:
        return {'error': 'No trades found'}
    
    # Overall performance
    initial_capital = df['capital'].iloc[0]
    final_capital = df['capital'].iloc[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    performance['initial_capital'] = initial_capital
    performance['final_capital'] = final_capital
    performance['total_return_pct'] = total_return
    performance['total_return_dollar'] = final_capital - initial_capital
    
    # Win rate and trade metrics
    num_trades = len(trades)
    profitable_trades = [trade for trade in trades if trade['pnl'] > 0]
    losing_trades = [trade for trade in trades if trade['pnl'] <= 0]
    win_rate = len(profitable_trades) / num_trades if num_trades > 0 else 0
    
    performance['num_trades'] = num_trades
    performance['win_rate'] = win_rate * 100  # as percentage
    
    # Average metrics
    if profitable_trades:
        avg_win = sum(trade['pnl'] for trade in profitable_trades) / len(profitable_trades)
        performance['avg_win'] = avg_win
    else:
        performance['avg_win'] = 0
    
    if losing_trades:
        avg_loss = sum(trade['pnl'] for trade in losing_trades) / len(losing_trades)
        performance['avg_loss'] = avg_loss
    else:
        performance['avg_loss'] = 0
    
    # Profit factor
    gross_profit = sum(trade['pnl'] for trade in profitable_trades)
    gross_loss = abs(sum(trade['pnl'] for trade in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    performance['gross_profit'] = gross_profit
    performance['gross_loss'] = gross_loss
    performance['profit_factor'] = profit_factor
    
    # Maximum drawdown
    running_max = df['equity'].iloc[0]
    drawdown = 0
    max_drawdown = 0
    peak_equity = running_max
    
    for equity in df['equity']:
        if equity > running_max:
            running_max = equity
        
        drawdown = (running_max - equity) / running_max * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            peak_equity = running_max
    
    performance['max_drawdown_pct'] = max_drawdown
    
    return performance

# Export dataframe to Excel
def export_to_excel(df, trades, performance, ticker="AAPL", filename=None):
    """Export dataframe to Excel file with trade list and performance metrics"""
    if filename is None:
        filename = f"{ticker}_backtest_results.xlsx"
        
    # Ensure file has .xlsx extension
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    # Create full path
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    # Remove existing file if it exists
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Create a writer to save multiple sheets
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        # Create a copy with timezone-naive index for data sheet
        df_export = df.copy()
        df_export.index = df_export.index.tz_localize(None)
        df_export.to_excel(writer, sheet_name='Data')
        
        # Create a trades sheet
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Convert any datetime columns to timezone-naive
            for col in trades_df.columns:
                if trades_df[col].dtype.kind == 'M':  # Check if column is datetime type
                    trades_df[col] = trades_df[col].dt.tz_localize(None)
            
            trades_df.to_excel(writer, sheet_name='Trades')
        
        # Create a performance sheet
        if performance:
            perf_df = pd.DataFrame(list(performance.items()), columns=['Metric', 'Value'])
            perf_df.to_excel(writer, sheet_name='Performance', index=False)
    
    print(f"Backtest results exported to {filepath}")
    
    return filepath

# Main function
def main():
    # Get AAPL data
    ticker = "AAPL"
    df = get_stock_data(ticker=ticker, period="30y")
    
    # Add indicators
    df = add_indicators(df, atr_length=14, dc_length_1=200, dc_length_2=100)
    
    # Add trading signals and run backtest
    df, trades, performance = add_trading_signals(df, initial_capital=100000, risk_pct=0.01)
    
    # Print latest values
    latest = df.iloc[-1]
    
    print(f"Latest data for {ticker} (as of {df.index[-1].date()}):")
    print(f"Price: ${latest['Close']:.2f}")
    print(f"ATR: {latest['atr']:.2f}")
    print(f"Signal: {int(latest['signal'])} ({latest['position_status']})")
    
    # Print performance summary
    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"Initial Capital: ${performance['initial_capital']:,.2f}")
    print(f"Final Capital: ${performance['final_capital']:,.2f}")
    print(f"Total Return: {performance['total_return_pct']:.2f}% (${performance['total_return_dollar']:,.2f})")
    print(f"Number of Trades: {performance['num_trades']}")
    print(f"Win Rate: {performance['win_rate']:.2f}%")
    print(f"Average Win: ${performance['avg_win']:,.2f}")
    print(f"Average Loss: ${performance['avg_loss']:,.2f}")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Maximum Drawdown: {performance['max_drawdown_pct']:.2f}%")
    
    # If currently in a position, show details
    if latest['signal'] != 0:
        direction = "LONG" if latest['signal'] == 1 else "SHORT"
        print(f"\nCurrent {direction} Position:")
        print(f"  Entry Price: ${latest['entry_price']:.2f}")
        print(f"  Current Price: ${latest['Close']:.2f}")
        print(f"  Shares: {int(latest['shares'])}")
        print(f"  Stop Loss: ${latest['stop_loss']:.2f}")
        pnl = latest['shares'] * (latest['Close'] - latest['entry_price']) if direction == "LONG" else \
              latest['shares'] * (latest['entry_price'] - latest['Close'])
        print(f"  Unrealized P&L: ${pnl:.2f}")
    
    # Export to Excel
    export_to_excel(df, trades, performance, ticker=ticker)

if __name__ == "__main__":
    main()