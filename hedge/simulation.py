import pandas as pd
import os
import numpy as np

def simulate_portfolio():
    signals_folder = './signals'
    simulation_folder = './simulation'
    os.makedirs(simulation_folder, exist_ok=True)
    
    # Load SPY signals
    spy_file = 'SPY_signals.csv'
    df = pd.read_csv(os.path.join(signals_folder, spy_file))
    
    # Initialize portfolio (cash only, no long SPY)
    initial_cash = 100000
    cash = initial_cash
    shares_short = 0
    in_position = False
    portfolio_values = [initial_cash]
    
    for i in range(1, len(df)):
        current_close = df.loc[i, 'close']
        signal = df.loc[i, 'signal']
        
        # Adjust positions based on signal
        if signal == 1 and not in_position:
            # Enter short: use 10% of current cash to short SPY
            cash_for_short = 0.1 * cash
            shares_to_short = cash_for_short / current_close
            cash += shares_to_short * current_close  # Shorting: cash increases from selling
            shares_short += shares_to_short
            in_position = True
        elif signal == 0 and in_position:
            # Exit short: buy back
            cash -= shares_short * current_close
            shares_short = 0
            in_position = False
        
        # Calculate current portfolio value (cash + short position)
        portfolio_value = cash - shares_short * current_close
        portfolio_values.append(portfolio_value)
    
    df['portfolio_value'] = portfolio_values
    df['daily_return'] = df['portfolio_value'].pct_change().fillna(0)
    
    # Calculate stats
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_cash) - 1
    num_days = len(portfolio_values) - 1
    trading_days = 252
    annualized_return = (final_value / initial_cash) ** (trading_days / num_days) - 1
    
    # Max drawdown
    cumulative = (1 + df['daily_return']).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assume risk-free rate = 0)
    mean_return = df['daily_return'].mean()
    std_return = df['daily_return'].std()
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    
    # Create stats DataFrame
    stats_df = pd.DataFrame({
        'Strategy': ['Short_SPY_Strategy'],
        'Final_Value': [final_value],
        'Total_Return': [total_return],
        'Annualized_Return': [annualized_return],
        'Max_Drawdown': [max_drawdown],
        'Sharpe_Ratio': [sharpe_ratio]
    })
    
    # Save stats
    stats_path = os.path.join(simulation_folder, 'strategy_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    
    # Save full simulation data
    sim_path = os.path.join(simulation_folder, 'SPY_simulation_full.csv')
    df.to_csv(sim_path, index=False)
    
    print(f'Simulation completed. Final portfolio value: ${final_value:.2f}. Stats saved to {stats_path}')

if __name__ == '__main__':
    simulate_portfolio()