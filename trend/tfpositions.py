import pandas as pd
import os
from classic import get_stock_data, add_indicators, add_trading_signals

def get_current_position(ticker, period="5y"):
    """Get the current trading position for a single asset"""
    # Get historical data
    df = get_stock_data(ticker=ticker, period=period)
    
    # Add indicators
    df = add_indicators(df, atr_length=25, dc_length_1=200, dc_length_2=100)
    
    # Add trading signals and run backtest
    df, trades, _ = add_trading_signals(df, initial_capital=100000, risk_pct=0.01)
    
    # Get latest position data
    latest = df.iloc[-1]
    
    # Create position data dictionary
    position_data = {
        'ticker': ticker,
        'date': df.index[-1],
        'close_price': latest['Close'],
        'signal': int(latest['signal']),  # 1=long, -1=short, 0=no position
        'position_status': latest['position_status'],
        'entry_date': None,
        'entry_price': None,
        'stop_loss': None,
        'atr': latest['atr']
    }
    
    # If in a position, add position details
    if latest['signal'] != 0:
        # Find when this position was entered
        signal_value = latest['signal']
        position_start_idx = None
        
        # Search backwards to find position entry
        for i in range(len(df)-2, -1, -1):
            if df['signal'].iloc[i] != signal_value:
                position_start_idx = i + 1
                break
        
        if position_start_idx is not None:
            entry_row = df.iloc[position_start_idx]
            position_data['entry_date'] = df.index[position_start_idx]
            position_data['entry_price'] = entry_row['entry_price']
            position_data['stop_loss'] = latest['stop_loss']  # Use current stop loss
    
    return position_data

def process_assets(tickers, period="5y"):
    """Process multiple assets and collect their current positions"""
    positions = []
    
    for ticker in tickers:
        try:
            print(f"Processing {ticker}...")
            position = get_current_position(ticker, period)
            positions.append(position)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
    return positions

def export_positions_to_excel(positions, filename="current_positions.xlsx"):
    """Export current positions to Excel"""
    if not positions:
        print("No positions data to export")
        return None
    
    # Create DataFrame from positions data
    positions_df = pd.DataFrame(positions)
    
    # Format position status for better readability
    positions_df['direction'] = positions_df['signal'].apply(
        lambda x: "LONG" if x == 1 else ("SHORT" if x == -1 else "NONE")
    )
    
    # Format dates if needed (make timezone naive)
    for col in positions_df.columns:
        if positions_df[col].dtype.kind == 'M':  # Check if column is datetime type
            positions_df[col] = pd.to_datetime(positions_df[col]).dt.tz_localize(None)
    
    # Reorder and select columns
    columns_order = [
        'ticker', 'direction', 'position_status', 'close_price', 
        'entry_date', 'entry_price', 'stop_loss', 'atr', 'date'
    ]
    
    # Keep only columns that exist
    columns_order = [col for col in columns_order if col in positions_df.columns]
    positions_df = positions_df[columns_order]
    
    # Create full path
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    # Remove existing file if it exists
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Export to Excel
    positions_df.to_excel(filepath, index=False)
    print(f"Positions exported to {filepath}")
    
    return filepath

def main():
    # Define list of assets to track
    tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
        "TSLA", "NVDA", "SPY", "QQQ", "GLD"
    ]
    
    # Process all assets
    positions = process_assets(tickers)
    
    # Export positions to Excel
    export_positions_to_excel(positions)

if __name__ == "__main__":
    main()