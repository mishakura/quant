import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def calculate_SUE(ticker_symbol):
    # Get ticker data
    ticker = yf.Ticker(ticker_symbol)
    
    # Get quarterly income statement data
    income_stmt = ticker.get_income_stmt(freq="quarterly")
    
    if income_stmt is None:
        return None, "No income statement data available"
    
    print(f"Number of quarters available: {len(income_stmt.columns)}")
    
    # Extract BasicEPS data
    try:
        eps_data = income_stmt.loc['BasicEPS']
        eps_data = eps_data.sort_index(ascending=False)  # Most recent first
        print("\nEPS Data by Quarter:")
        print(eps_data)
    except KeyError:
        return None, "BasicEPS data not available"
    
    # First check if we have minimum 6 quarters
    if len(eps_data) < 6:
        return None, f"Insufficient quarters (found {len(eps_data)}, need 6)"
    
    # Calculate year-over-year changes
    eps_changes = []
    print("\nCalculating year-over-year changes:")
    for i in range(len(eps_data)-4):  # -4 to ensure we can look back 4 quarters
        current_eps = eps_data.iloc[i]
        year_ago_eps = eps_data.iloc[i+4]
        print(f"Current EPS: {current_eps}, Year ago EPS: {year_ago_eps}")
        if pd.isna(current_eps) or pd.isna(year_ago_eps):
            print(f"Skipping due to NaN values")
            continue
        change = current_eps - year_ago_eps
        eps_changes.append(change)
        print(f"Change: {change}")
    
    print(f"\nNumber of valid year-over-year changes: {len(eps_changes)}")
    
    # Need at least 2 changes (which requires 6 quarters of data)
    if len(eps_changes) < 2:
        return None, f"Insufficient year-over-year changes (found {len(eps_changes)}, need 2)"
    
    # Calculate most recent change
    latest_change = eps_changes[0]
    
    # Calculate standard deviation of last 8 changes (or all available if less than 8)
    lookback = min(8, len(eps_changes))
    std_dev = np.std(eps_changes[:lookback])
    
    # Calculate SUE
    if std_dev != 0:
        sue = latest_change / std_dev
    else:
        return None, "Zero standard deviation in earnings changes"
    
    return sue, "Success"

def create_sue_results():
    # List of stocks to analyze (you can modify this list)
    tickers = ['NFLX']
    
    # Create a list to store results
    results = []
    calculation_date = datetime.now().strftime('%Y-%m-%d')
    
    for ticker in tickers:
        sue_value, status = calculate_SUE(ticker)
        
        result = {
            'Ticker': ticker,
            'Calculation_Date': calculation_date,
            'SUE_Value': sue_value if sue_value is not None else None,
            'Status': status
        }
        results.append(result)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_filename = f'SUE_results_{calculation_date}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    return df

# Run the analysis and save to CSV
if __name__ == "__main__":
    results_df = create_sue_results()
    print("\nResults Preview:")
    print(results_df)