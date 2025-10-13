import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from pathlib import Path
import scipy.stats

# Import classic strategy module
from classic import add_indicators, calculate_performance

class PortfolioBacktest:
    def __init__(self, initial_capital=100000, risk_pct=0.005):
        """
        Initialize portfolio backtest with starting capital and risk percentage
        
        Args:
            initial_capital (float): Starting capital in USD
            risk_pct (float): Risk per trade as percentage of capital (0.005 = 0.5%)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_pct = risk_pct
        self.positions = {}  # Current open positions {symbol: {contracts, entry_price, direction, entry_date, atr}}
        self.trades_history = []
        self.equity_curve = []
        self.dates = []
        self.symbols = []  # List of symbols in alphabetical order
        self.data_dict = {}  # Dictionary to store data for each symbol
        
    def load_data(self, symbols, period="5y", interval="1d"):
        """
        Load data for all symbols
        
        Args:
            symbols (list): List of stock symbols
            period (str): Period to download data for
            interval (str): Data interval
        """
        print(f"Loading data for {len(symbols)} symbols...")
        self.symbols = sorted(symbols)  # Sort symbols alphabetically
        
        for symbol in self.symbols:
            try:
                print(f"Loading data for {symbol}...")
                stock = yf.Ticker(symbol)
                df = stock.history(period=period, interval=interval)
                
                # Ensure required columns exist
                if 'Open' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
                    print(f"Missing required price columns for {symbol}, skipping")
                    continue
                    
                # Reset index to make date a column
                df = df.reset_index()
                
                # Add indicators (ATR and Donchian Channels)
                df = add_indicators(df)
                
                # Add signal columns
                df['long_entry_signal'] = False
                df['long_exit_signal'] = False
                df['short_entry_signal'] = False
                df['short_exit_signal'] = False
                df['signal'] = 0
                
                # Generate entry/exit signals based on Donchian Channel breakouts
                for i in range(1, len(df)):
                    prev_idx = i - 1
                    
                    # Check conditions against previous day's Donchian values
                    long_entry = df['Close'].iloc[i] > df['dc_upper_200'].iloc[prev_idx]
                    long_exit = df['Close'].iloc[i] < df['dc_lower_100'].iloc[prev_idx]
                    short_entry = df['Close'].iloc[i] < df['dc_lower_200'].iloc[prev_idx]
                    short_exit = df['Close'].iloc[i] > df['dc_upper_100'].iloc[prev_idx]
                    
                    df.loc[df.index[i], 'long_entry_signal'] = long_entry
                    df.loc[df.index[i], 'long_exit_signal'] = long_exit
                    df.loc[df.index[i], 'short_entry_signal'] = short_entry
                    df.loc[df.index[i], 'short_exit_signal'] = short_exit
                
                self.data_dict[symbol] = df
                print(f"Successfully loaded data for {symbol} with {len(df)} rows")
            except Exception as e:
                print(f"Error loading data for {symbol}: {e}")
        
        print(f"Successfully loaded data for {len(self.data_dict)} symbols")
        
        # Identify all unique dates across all dataframes
        all_dates = set()
        for symbol, df in self.data_dict.items():
            all_dates.update(df['Date'].tolist())
        
        self.all_dates = sorted(all_dates)
        print(f"Backtest will run from {min(self.all_dates)} to {max(self.all_dates)}")
        
    def calculate_position_size(self, price, atr):
        """
        Calculate position size based on ATR and risk percentage
        
        Args:
            price (float): Current asset price
            atr (float): Average True Range value
            
        Returns:
            tuple: (contracts, usd_position)
        """
        if atr == 0 or np.isnan(atr):
            return 0, 0
            
        risk_amount = self.capital * self.risk_pct
        contracts = risk_amount / atr
        usd_position = contracts * price
        
        # Ensure position doesn't exceed available capital
        if usd_position > self.capital:
            contracts = self.capital / price
            usd_position = self.capital
            
        return contracts, usd_position
    
    def process_signals_for_date(self, current_date):
        """
        Process trading signals for all symbols on a specific date
        
        Args:
            current_date: Date to process signals for
        """
        daily_pnl = 0
        
        # Process each symbol in alphabetical order
        for symbol in self.symbols:
            if symbol not in self.data_dict:
                continue
                
            df = self.data_dict[symbol]
            
            # Check if we have data for this date
            date_rows = df[df['Date'] == current_date]
            if len(date_rows) == 0:
                continue
                
            # Get current row
            current_row = date_rows.iloc[0]
            current_idx = current_row.name
            
            # Skip if not enough history
            if current_idx < 2:
                continue
                
            # Get previous row for signal checking
            prev_idx = current_idx - 1
            prev_row = df.iloc[prev_idx]
            
            current_price = current_row['Open']  # Use open price for execution
            current_atr = current_row['atr']
            
            # Check for exits first (close existing positions)
            if symbol in self.positions:
                position = self.positions[symbol]
                exit_signal = False
                exit_reason = None
                
                # Check stop loss
                if position['direction'] == 1 and current_price < position['stop_loss']:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                elif position['direction'] == -1 and current_price > position['stop_loss']:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                
                # Check Donchian exit
                if position['direction'] == 1 and prev_row['long_exit_signal']:
                    exit_signal = True
                    exit_reason = "Donchian Exit"
                elif position['direction'] == -1 and prev_row['short_exit_signal']:
                    exit_signal = True
                    exit_reason = "Donchian Exit"
                
                # Exit position if signal detected
                if exit_signal:
                    contracts = position['contracts']
                    entry_price = position['entry_price']
                    direction = position['direction']
                    
                    # Calculate P&L
                    if direction == 1:  # Long position
                        pnl = contracts * (current_price - entry_price)
                    else:  # Short position
                        pnl = contracts * (entry_price - current_price)
                        
                    daily_pnl += pnl
                    self.capital += pnl
                    
                    # Record trade
                    self.trades_history.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'contracts': contracts,
                        'direction': "LONG" if direction == 1 else "SHORT",
                        'pnl': pnl,
                        'pnl_pct': (pnl / (contracts * entry_price)) * 100 if contracts * entry_price > 0 else 0,
                        'exit_reason': exit_reason
                    })
                    
                    # Remove position
                    del self.positions[symbol]
                    
                    print(f"{current_date}: Closed {direction} position in {symbol} at {current_price:.2f}, P&L: {pnl:.2f} ({exit_reason})")
            
            # Check for new entries
            if symbol not in self.positions:
                # Long entry signal
                if prev_row['long_entry_signal']:
                    contracts, position_value = self.calculate_position_size(current_price, current_atr)
                    
                    if contracts > 0 and position_value > 0:
                        # Calculate stop loss
                        stop_loss = current_price - (current_atr * 3)
                        
                        # Record position
                        self.positions[symbol] = {
                            'contracts': contracts,
                            'entry_price': current_price,
                            'direction': 1,  # Long
                            'entry_date': current_date,
                            'atr': current_atr,
                            'stop_loss': stop_loss
                        }
                        
                        print(f"{current_date}: Opened LONG position in {symbol} at {current_price:.2f}, Contracts: {contracts:.2f}, Stop: {stop_loss:.2f}")
                
                # Short entry signal
                elif prev_row['short_entry_signal']:
                    contracts, position_value = self.calculate_position_size(current_price, current_atr)
                    
                    if contracts > 0 and position_value > 0:
                        # Calculate stop loss
                        stop_loss = current_price + (current_atr * 3)
                        
                        # Record position
                        self.positions[symbol] = {
                            'contracts': contracts,
                            'entry_price': current_price,
                            'direction': -1,  # Short
                            'entry_date': current_date,
                            'atr': current_atr,
                            'stop_loss': stop_loss
                        }
                        
                        print(f"{current_date}: Opened SHORT position in {symbol} at {current_price:.2f}, Contracts: {contracts:.2f}, Stop: {stop_loss:.2f}")
        
        # Record daily portfolio value
        portfolio_value = self.capital
        
        # Add unrealized P&L from open positions
        for symbol, position in self.positions.items():
            # Get latest price for this symbol on this date
            df = self.data_dict[symbol]
            date_rows = df[df['Date'] == current_date]
            
            if len(date_rows) > 0:
                latest_price = date_rows.iloc[0]['Close']  # Use close for valuation
                contracts = position['contracts']
                entry_price = position['entry_price']
                direction = position['direction']
                
                # Calculate unrealized P&L
                if direction == 1:  # Long position
                    unrealized_pnl = contracts * (latest_price - entry_price)
                else:  # Short position
                    unrealized_pnl = contracts * (entry_price - latest_price)
                    
                portfolio_value += unrealized_pnl
        
        # Record equity and date
        self.equity_curve.append(portfolio_value)
        self.dates.append(current_date)
        
        # Return daily portfolio value
        return portfolio_value
    
    def run_backtest(self):
        """
        Run the portfolio backtest
        """
        print("Starting portfolio backtest...")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Risk percentage: {self.risk_pct*100:.2f}%")
        print(f"Processing {len(self.symbols)} symbols in alphabetical order")
        
        # Record starting equity
        self.equity_curve = [self.initial_capital]
        
        # Process each date
        for date_idx, current_date in enumerate(self.all_dates):
            if date_idx % 100 == 0:
                print(f"Processing date {date_idx+1}/{len(self.all_dates)}: {current_date}")
                
            portfolio_value = self.process_signals_for_date(current_date)
    
    def calculate_statistics(self):
        """
        Calculate performance statistics
        
        Returns:
            dict: Performance statistics
        """
        trades_df = pd.DataFrame(self.trades_history)
        equity_curve = np.array(self.equity_curve)
        
        # Handle case with no trades
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'kurtosis': 0,
                'skewness': 0,
                'max_win_usd': 0,
                'start_date': self.all_dates[0] if self.all_dates else None
            }
            
        # Trade statistics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        total_trades = len(trades_df)
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Maximum win in USD terms
        max_win_usd = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')
        
        # Portfolio statistics
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        
        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate kurtosis and skewness of returns
        kurtosis = scipy.stats.kurtosis(returns) if len(returns) > 3 else 0
        skewness = scipy.stats.skew(returns) if len(returns) > 2 else 0
        
        # Assuming 252 trading days in a year
        trading_days = len(returns)
        annualized_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Risk metrics
        daily_returns = returns
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        excess_return = annualized_return - risk_free_rate
        
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Downside deviation (for Sortino)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Start date of the system
        start_date = self.all_dates[0] if self.all_dates else None
        
        # Calculate metrics by symbol
        symbol_stats = {}
        for symbol in self.symbols:
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            if len(symbol_trades) > 0:
                winning = symbol_trades[symbol_trades['pnl'] > 0]
                win_rate_symbol = len(winning) / len(symbol_trades) if len(symbol_trades) > 0 else 0
                
                symbol_stats[symbol] = {
                    'trades': len(symbol_trades),
                    'win_rate': win_rate_symbol,
                    'net_pnl': symbol_trades['pnl'].sum(),
                    'avg_pnl': symbol_trades['pnl'].mean()
                }
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades_count,
            'losing_trades': losing_trades_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win_usd': max_win_usd,  # Added max win
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'kurtosis': kurtosis,  # Added kurtosis
            'skewness': skewness,  # Added skewness
            'start_date': start_date,  # Added start date
            'symbol_stats': symbol_stats,
            'initial_capital': self.initial_capital,
            'final_capital': equity_curve[-1],
            'total_return_dollar': equity_curve[-1] - self.initial_capital
        }
    
    def export_to_excel(self, filename="portfolio_backtest_results.xlsx"):
        """
        Export backtest results to Excel
        
        Args:
            filename (str): Output Excel filename
        """
        # Create Excel writer
        print(f"Exporting backtest results to {filename}...")
        
        # Calculate statistics
        stats = self.calculate_statistics()
        
        # Ensure start_date is timezone-naive
        if stats['start_date'] is not None and hasattr(stats['start_date'], 'tz_localize'):
            start_date = stats['start_date'].tz_localize(None)
        else:
            start_date = stats['start_date']
        
        # Helper function to convert datetime columns to timezone-naive
        def convert_timezones(df):
            for col in df.columns:
                if pd.api.types.is_datetime64tz_dtype(df[col]):
                    df[col] = df[col].dt.tz_localize(None)
            return df
        
        with pd.ExcelWriter(filename) as writer:
            # Summary statistics
            stats_df = pd.DataFrame({
                'Metric': [
                    'Initial Capital',
                    'Final Capital',
                    'Total Return ($)',
                    'Total Return (%)',
                    'Annualized Return (%)',
                    'Total Trades',
                    'Winning Trades',
                    'Losing Trades',
                    'Win Rate (%)',
                    'Average Win ($)',
                    'Average Loss ($)',
                    'Maximum Win ($)',  # Added max win
                    'Profit Factor',
                    'Maximum Drawdown (%)',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Kurtosis',  # Added kurtosis
                    'Skewness',  # Added skewness
                    'Start Date'  # Added start date
                ],
                'Value': [
                    f"${stats['initial_capital']:,.2f}",
                    f"${stats['final_capital']:,.2f}",
                    f"${stats['total_return_dollar']:,.2f}",
                    f"{stats['total_return']*100:.2f}%",
                    f"{stats['annualized_return']*100:.2f}%",
                    stats['total_trades'],
                    stats['winning_trades'],
                    stats['losing_trades'],
                    f"{stats['win_rate']*100:.2f}%",
                    f"${stats['avg_win']:,.2f}",
                    f"${stats['avg_loss']:,.2f}",
                    f"${stats['max_win_usd']:,.2f}",  # Added max win
                    f"{stats['profit_factor']:.2f}",
                    f"{stats['max_drawdown']*100:.2f}%",
                    f"{stats['sharpe_ratio']:.2f}",
                    f"{stats['sortino_ratio']:.2f}",
                    f"{stats['kurtosis']:.4f}",  # Added kurtosis
                    f"{stats['skewness']:.4f}",  # Added skewness
                    start_date  # Use the timezone-naive start date
                ]
            })
            stats_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Trades history
            trades_df = pd.DataFrame(self.trades_history)
            if not trades_df.empty:
                # Convert any timezone-aware datetime columns
                trades_df = convert_timezones(trades_df)
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
            
            # Equity curve
            equity_df = pd.DataFrame({
                'Date': self.dates,
                'Equity': self.equity_curve[1:]  # Skip the first value which is initial capital
            })
            # Convert any timezone-aware datetime columns
            equity_df = convert_timezones(equity_df)
            equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
            
            # Symbol statistics
            if 'symbol_stats' in stats and stats['symbol_stats']:
                symbol_stats_data = []
                for symbol, symbol_data in stats['symbol_stats'].items():
                    symbol_stats_data.append({
                        'Symbol': symbol,
                        'Trades': symbol_data['trades'],
                        'Win Rate': f"{symbol_data['win_rate']*100:.2f}",
                        'Net P&L': f"${symbol_data['net_pnl']:,.2f}",
                        'Avg P&L': f"${symbol_data['avg_pnl']:,.2f}"
                    })
                
                symbol_stats_df = pd.DataFrame(symbol_stats_data)
                symbol_stats_df.to_excel(writer, sheet_name='Symbol Stats', index=False)
            
            # Positions at the end
            positions_df = pd.DataFrame([{
                'Symbol': symbol,
                'Contracts': pos['contracts'],
                'Entry Price': pos['entry_price'],
                'Current Direction': 'Long' if pos['direction'] == 1 else 'Short',
                'Entry Date': pos['entry_date'],
                'Stop Loss': pos['stop_loss']
            } for symbol, pos in self.positions.items()])
            
            if not positions_df.empty:
                positions_df = convert_timezones(positions_df)
                positions_df.to_excel(writer, sheet_name='Open Positions', index=False)
        
        print(f"Backtest results exported to {filename}")
        
        # Create equity curve chart
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, self.equity_curve[1:], label='Portfolio Equity')
        plt.title('Portfolio Backtest Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save chart
        chart_filename = filename.replace('.xlsx', '_equity_curve.png')
        plt.savefig(chart_filename)
        plt.close()
        
        print(f"Equity curve chart saved to {chart_filename}")


def main():
    # List of symbols to include in the portfolio
    # Using some popular stocks as an example
    symbols = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM',
        'V', 'WMT', 'PG', 'DIS', 'NFLX', 'INTC', 'AMD', 'ADBE'
    ]
    
    # Initialize backtest with $100,000 and 0.5% risk
    backtest = PortfolioBacktest(initial_capital=100000, risk_pct=0.005)
    
    # Load data for all symbols (5 years of history)
    backtest.load_data(symbols, period="5y")
    
    # Run the backtest
    backtest.run_backtest()
    
    # Export results to Excel
    backtest.export_to_excel("classic_portfolio_backtest.xlsx")
    
    # Print summary statistics
    stats = backtest.calculate_statistics()
    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"Initial Capital: ${stats['initial_capital']:,.2f}")
    print(f"Final Capital: ${stats['final_capital']:,.2f}")
    print(f"Total Return: {stats['total_return']*100:.2f}% (${stats['total_return_dollar']:,.2f})")
    print(f"Annualized Return: {stats['annualized_return']*100:.2f}%")
    print(f"Number of Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']*100:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Maximum Drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {stats['sortino_ratio']:.2f}")
    

if __name__ == "__main__":
    main()