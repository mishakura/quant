import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class DynamicHedgeBacktest:
    def __init__(self, spx_data_path, gc_data_path, ema_span=50, spx_hedge_pct=50, gc_hedge_pct=25, cash_hedge_pct=25):
        """
        Initialize the backtester with SPX and GC data
        
        Parameters:
        spx_data_path: path to SPX CSV file
        gc_data_path: path to GC (Gold) CSV file
        ema_span: EMA span/period (default 50)
        spx_hedge_pct: % of portfolio in SPX when hedged (default 50)
        gc_hedge_pct: % of portfolio in GC when hedged (default 25)
        cash_hedge_pct: % of portfolio in cash when hedged (default 25)
        """
        self.spx_df = pd.read_csv(spx_data_path, parse_dates=['Date'])
        self.gc_df = pd.read_csv(gc_data_path, parse_dates=['Date'])
        self.ema_span = ema_span
        self.spx_hedge_pct = spx_hedge_pct / 100  # Convert to decimal
        self.gc_hedge_pct = gc_hedge_pct / 100
        self.cash_hedge_pct = cash_hedge_pct / 100
        
        # Validate allocations sum to 100%
        total_allocation = self.spx_hedge_pct + self.gc_hedge_pct + self.cash_hedge_pct
        if not np.isclose(total_allocation, 1.0):
            raise ValueError(f"Allocations must sum to 100%. Current sum: {total_allocation * 100:.2f}%")
        
        # Merge data on date
        self.data = self.spx_df.merge(self.gc_df, on='Date', suffixes=('_SPX', '_GC'))
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
    def calculate_indicators(self):
        """Calculate EMA indicator"""
        # Calculate EMA on close prices
        self.data[f'ema_{self.ema_span}'] = self.data['Close_SPX'].ewm(span=self.ema_span, adjust=False).mean()
        
        # Shift by 1 to avoid look-ahead bias
        self.data[f'prev_ema_{self.ema_span}'] = self.data[f'ema_{self.ema_span}'].shift(1)
        
        # Debug: print first few valid indicators
        print("\n=== INDICATOR DEBUG ===")
        print(f"Using EMA span: {self.ema_span}")
        print(f"Hedge Allocation - SPX: {self.spx_hedge_pct*100:.1f}%, GC: {self.gc_hedge_pct*100:.1f}%, Cash: {self.cash_hedge_pct*100:.1f}%")
        valid_idx = self.data[f'prev_ema_{self.ema_span}'].first_valid_index()
        if valid_idx is not None:
            for i in range(valid_idx, min(valid_idx + 10, len(self.data))):
                print(f"Date: {self.data.loc[i, 'Date']}, Close: {self.data.loc[i, 'Close_SPX']:.2f}, "
                      f"Prev EMA {self.ema_span}: {self.data.loc[i, f'prev_ema_{self.ema_span}']:.2f}, "
                      f"Below EMA: {self.data.loc[i, 'Close_SPX'] < self.data.loc[i, f'prev_ema_{self.ema_span}']}")
        
    def run_dynamic_hedge_strategy(self, initial_capital=100000):
        """
        Run the dynamic hedge strategy with EMA
        
        Returns portfolio value series
        """
        portfolio_value = []
        cash = 0
        spx_position = initial_capital  # Start 100% in SPX
        gc_position = 0
        
        position_state = 'long_spx'  # 'long_spx' or 'hedged'
        
        trades = []  # Track trades for debugging
        
        print("\n=== TRADE SIGNAL DEBUG ===")
        
        for idx, row in self.data.iterrows():
            if pd.isna(row[f'prev_ema_{self.ema_span}']):
                portfolio_value.append(np.nan)
                continue
                
            # Calculate current portfolio value based on price changes
            if idx > 0 and pd.notna(self.data.loc[idx-1, 'Close_SPX']):
                spx_return = row['Close_SPX'] / self.data.loc[idx-1, 'Close_SPX']
                gc_return = row['Close_GC'] / self.data.loc[idx-1, 'Close_GC']
                
                spx_position = spx_position * spx_return
                if gc_position > 0:
                    gc_position = gc_position * gc_return
            
            total_value = spx_position + gc_position + cash
            portfolio_value.append(total_value)
            
            # Check for signals - compare today's close with previous day's EMA
            current_close = row['Close_SPX']
            prev_ema = row[f'prev_ema_{self.ema_span}']
            
            if position_state == 'long_spx':
                # Check if SPX closes below previous EMA
                if current_close < prev_ema:
                    # Rebalance to hedge allocation
                    total_portfolio = spx_position
                    cash = total_portfolio * self.cash_hedge_pct
                    gc_position = total_portfolio * self.gc_hedge_pct
                    spx_position = total_portfolio * self.spx_hedge_pct
                    position_state = 'hedged'
                    trades.append({
                        'date': row['Date'],
                        'action': 'HEDGE',
                        'spx_close': current_close,
                        'ema': prev_ema,
                        'spx_pct': self.spx_hedge_pct * 100,
                        'gc_pct': self.gc_hedge_pct * 100,
                        'cash_pct': self.cash_hedge_pct * 100
                    })
                    print(f"*** HEDGE TRIGGERED on {row['Date']}: SPX Close {current_close:.2f} < EMA {self.ema_span} {prev_ema:.2f} ***")
                    print(f"    Allocation: {self.spx_hedge_pct*100:.1f}% SPX, {self.gc_hedge_pct*100:.1f}% GC, {self.cash_hedge_pct*100:.1f}% Cash")
                    
            elif position_state == 'hedged':
                # Check if SPX closes above previous EMA
                if current_close > prev_ema:
                    # Go back to 100% SPX
                    total_portfolio = spx_position + gc_position + cash
                    spx_position = total_portfolio
                    gc_position = 0
                    cash = 0
                    position_state = 'long_spx'
                    trades.append({
                        'date': row['Date'],
                        'action': 'EXIT HEDGE',
                        'spx_close': current_close,
                        'ema': prev_ema
                    })
                    print(f"*** EXIT HEDGE on {row['Date']}: SPX Close {current_close:.2f} > EMA {self.ema_span} {prev_ema:.2f} ***")
                    print(f"    Back to 100% SPX")
        
        self.data['dynamic_hedge_value'] = portfolio_value
        self.trades = trades
        print(f"\nTotal trades executed: {len(trades)}")
        return self.data['dynamic_hedge_value']
    
    def run_buy_hold_strategy(self, initial_capital=100000):
        """
        Run simple buy and hold SPX strategy
        
        Returns portfolio value series
        """
        portfolio_value = []
        current_value = None
        
        for idx, row in self.data.iterrows():
            if pd.isna(row[f'prev_ema_{self.ema_span}']):
                portfolio_value.append(np.nan)
            else:
                if current_value is None:
                    # Initialize at first valid index
                    current_value = initial_capital
                    portfolio_value.append(current_value)
                else:
                    prev_idx = idx - 1
                    if pd.notna(self.data.loc[prev_idx, 'Close_SPX']):
                        price_return = row['Close_SPX'] / self.data.loc[prev_idx, 'Close_SPX']
                        current_value = current_value * price_return
                        portfolio_value.append(current_value)
                    else:
                        portfolio_value.append(current_value)
        
        self.data['buy_hold_value'] = portfolio_value
        return self.data['buy_hold_value']
    
    def calculate_statistics(self):
        """Calculate performance statistics for both strategies"""
        
        # Drop NaN values for calculations
        valid_data = self.data.dropna(subset=['dynamic_hedge_value', 'buy_hold_value'])
        
        stats = {}
        
        for strategy in ['dynamic_hedge', 'buy_hold']:
            col_name = f'{strategy}_value'
            
            # Returns
            returns = valid_data[col_name].pct_change().dropna()
            
            # Total return
            total_return = (valid_data[col_name].iloc[-1] / valid_data[col_name].iloc[0] - 1) * 100
            
            # Calculate actual years from dates
            start_date = valid_data['Date'].iloc[0]
            end_date = valid_data['Date'].iloc[-1]
            years = (end_date - start_date).days / 365.25
            
            # Annualized return
            annualized_return = (((valid_data[col_name].iloc[-1] / valid_data[col_name].iloc[0]) ** (1/years)) - 1) * 100
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe = (annualized_return / volatility) if volatility > 0 else 0
            
            # Maximum drawdown
            cummax = valid_data[col_name].expanding().max()
            drawdown = (valid_data[col_name] - cummax) / cummax
            max_drawdown = drawdown.min() * 100
            
            # Calmar ratio
            calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Win rate (percentage of positive return days)
            win_rate = (returns > 0).sum() / len(returns) * 100
            
            # Number of trades (for dynamic hedge only)
            num_trades = len(self.trades) if strategy == 'dynamic_hedge' and hasattr(self, 'trades') else 0
            
            stats[strategy] = {
                'Total Return (%)': total_return,
                'Annualized Return (%)': annualized_return,
                'Annualized Volatility (%)': volatility,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': max_drawdown,
                'Calmar Ratio': calmar,
                'Win Rate (%)': win_rate,
                'Number of Trades': num_trades if strategy == 'dynamic_hedge' else 'N/A',
                'Years': years,
                'Final Value ($)': valid_data[col_name].iloc[-1],
                'Start Date': start_date.strftime('%Y-%m-%d'),
                'End Date': end_date.strftime('%Y-%m-%d')
            }
        
        return stats
    
    def plot_results(self):
        """Plot the performance comparison"""
        valid_data = self.data.dropna(subset=['dynamic_hedge_value', 'buy_hold_value'])
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Portfolio value over time
        axes[0].plot(valid_data['Date'], valid_data['dynamic_hedge_value'], 
                     label=f'Dynamic Hedge ({self.spx_hedge_pct*100:.0f}% SPX, {self.gc_hedge_pct*100:.0f}% GC, {self.cash_hedge_pct*100:.0f}% Cash)', linewidth=2)
        axes[0].plot(valid_data['Date'], valid_data['buy_hold_value'], 
                     label='Buy & Hold SPX', linewidth=2, alpha=0.7)
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title(f'Portfolio Value Comparison (EMA {self.ema_span})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown comparison
        for strategy, label in [('dynamic_hedge_value', 'Dynamic Hedge'), 
                                 ('buy_hold_value', 'Buy & Hold')]:
            cummax = valid_data[strategy].expanding().max()
            drawdown = (valid_data[strategy] - cummax) / cummax * 100
            axes[1].plot(valid_data['Date'], drawdown, label=label, linewidth=2)
        
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_title('Drawdown Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(valid_data['Date'], 0, axes[1].get_ylim()[0], alpha=0.1, color='red')
        
        # SPX price with EMA
        axes[2].plot(valid_data['Date'], valid_data['Close_SPX'], 
                     label='SPX Close', linewidth=1.5, color='black')
        axes[2].plot(valid_data['Date'], valid_data[f'ema_{self.ema_span}'], 
                     label=f'{self.ema_span} EMA', linewidth=1, linestyle='--', alpha=0.7, color='red')
        axes[2].set_ylabel('SPX Price')
        axes[2].set_xlabel('Date')
        axes[2].set_title(f'SPX with {self.ema_span} EMA')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'backtest_results_ema_{self.ema_span}_spx{self.spx_hedge_pct*100:.0f}_gc{self.gc_hedge_pct*100:.0f}_cash{self.cash_hedge_pct*100:.0f}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_full_backtest(self):
        """Run complete backtest and display results"""
        print("Calculating indicators...")
        self.calculate_indicators()
        
        print("Running Dynamic Hedge Strategy...")
        self.run_dynamic_hedge_strategy()
        
        print("Running Buy & Hold Strategy...")
        self.run_buy_hold_strategy()
        
        print("\nCalculating statistics...")
        stats = self.calculate_statistics()
        
        # Display results
        print("\n" + "="*80)
        print(f"BACKTEST RESULTS (EMA {self.ema_span})")
        print(f"Hedge Allocation: {self.spx_hedge_pct*100:.1f}% SPX | {self.gc_hedge_pct*100:.1f}% GC | {self.cash_hedge_pct*100:.1f}% Cash")
        print("="*80)
        
        for strategy in ['dynamic_hedge', 'buy_hold']:
            strategy_name = 'DYNAMIC HEDGE STRATEGY' if strategy == 'dynamic_hedge' else 'BUY & HOLD SPX'
            print(f"\n{strategy_name}")
            print("-"*80)
            for metric, value in stats[strategy].items():
                if isinstance(value, float):
                    print(f"{metric:.<40} {value:>15,.2f}")
                else:
                    print(f"{metric:.<40} {value:>15}")
        
        print("\n" + "="*80)
        print("\nPlotting results...")
        self.plot_results()
        
        return stats


# Example usage:
if __name__ == "__main__":
    # Replace with your actual file paths
    SPX_DATA = "SPX.csv"
    GC_DATA = "GC.csv"
    
    # Configurable parameters
    EMA_SPAN = 200  # Try 50, 100, 200, etc.
    SPX_HEDGE_PCT = 80  # % in SPX when hedged
    GC_HEDGE_PCT = 10  # % in GC when hedged
    CASH_HEDGE_PCT = 10 # % in cash when hedged
    # Note: SPX_HEDGE_PCT + GC_HEDGE_PCT + CASH_HEDGE_PCT must equal 100
    
    backtest = DynamicHedgeBacktest(
        SPX_DATA, 
        GC_DATA, 
        ema_span=EMA_SPAN,
        spx_hedge_pct=SPX_HEDGE_PCT,
        gc_hedge_pct=GC_HEDGE_PCT,
        cash_hedge_pct=CASH_HEDGE_PCT
    )
    results = backtest.run_full_backtest()