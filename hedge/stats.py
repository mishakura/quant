import pandas as pd
import numpy as np
import os

# Paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
indicators_dir = os.path.join(os.path.dirname(__file__), 'indicators')
vix_file = os.path.join(indicators_dir, 'VIX.csv')
spy_file = os.path.join(data_dir, 'SPY.csv')

# Load VIX data
vix_df = pd.read_csv(vix_file)
vix_df['Date'] = pd.to_datetime(vix_df['observation_date'])
vix_df = vix_df.set_index('Date')

# Load SPY data
spy_df = pd.read_csv(spy_file)
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_df = spy_df.set_index('Date')

# Compute daily returns - for strategy, use 'Capital' as the portfolio value
ret_strategy = vix_df['Capital'].pct_change().dropna()
ret_spy = spy_df['Close'].pct_change().dropna()

# Align dates to common period
common_dates = ret_strategy.index.intersection(ret_spy.index)
ret_strategy = ret_strategy.loc[common_dates]
ret_spy = ret_spy.loc[common_dates]

# Function to calculate stats
def calculate_stats(returns, benchmark_returns=None, rf=0.03):
    n = len(returns)
    if n == 0:
        return {}
    
    ann_return = (1 + returns).prod() ** (252 / n) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252 - rf) / ann_vol if ann_vol > 0 else np.nan
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    # Beta
    if benchmark_returns is not None:
        aligned_bench = benchmark_returns.reindex(returns.index).dropna()
        aligned_ret = returns.loc[aligned_bench.index]
        if len(aligned_ret) > 1:
            cov = np.cov(aligned_ret, aligned_bench)[0, 1]
            var_bench = aligned_bench.var()
            beta = cov / var_bench if var_bench > 0 else np.nan
        else:
            beta = np.nan
    else:
        beta = 1.0  # For SPY itself
    
    # Other stats
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    win_rate = (returns > 0).mean()
    total_return = (1 + returns).prod() - 1
    
    return {
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Beta': beta,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Win Rate': win_rate,
        'Total Return': total_return
    }

# Calculate stats for SPY
spy_stats = calculate_stats(ret_spy, rf=0.03)

# Calculate stats for Strategy
strategy_stats = calculate_stats(ret_strategy, benchmark_returns=ret_spy, rf=0.03)

# Output to Excel
with pd.ExcelWriter('stats.xlsx') as writer:
    pd.DataFrame([spy_stats], index=['SPY']).T.to_excel(writer, sheet_name='SPY_Stats')
    pd.DataFrame([strategy_stats], index=['Strategy']).T.to_excel(writer, sheet_name='Strategy_Stats')

print("Stats calculated and saved to stats.xlsx")