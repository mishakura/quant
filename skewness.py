import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from tickers import tickers
def compute_rolling_skewness(prices, window=252):
    """Compute rolling skewness using a single rolling window operation"""
    # Compute daily returns
    returns = prices.pct_change()
    
    # Use a single rolling window calculation for skewness
    # using the scipy.stats formula implemented in pandas
    # This will directly compute skewness over the rolling window
    # preventing the compounding of NaN values from multiple rolling operations
    rolling_skew = returns.rolling(window=window, min_periods=window).apply(
        lambda x: stats.skew(x, nan_policy='omit'), raw=True)
    
    return rolling_skew

def calculate_portfolio_weights(skewness_values):
    """
    Calculate portfolio weights following Asness et al. (2013) and Koijen et al. (2018).
    Assets with more negative skewness receive higher weights.
    
    Formula: w_i,t = z_t · (rank(-1 × Skew_i,t) - (N_t + 1)/2)
    
    Long positions sum to 1 and short positions sum to -1
    
    Returns:
        weights: Series of portfolio weights
        ranks: Series of ranks (higher rank = more negative skewness)
    """
    # Count available assets
    N_t = len(skewness_values.dropna())
    
    if N_t <= 1:
        print("At least 2 assets needed for ranking. Not enough valid skewness values.")
        return pd.Series(0, index=skewness_values.index), pd.Series(0, index=skewness_values.index)
    
    # Rank the negative skewness (-1 × Skew)
    # Higher rank = more negative skewness
    ranks = (-1 * skewness_values).rank()
    
    # Calculate the de-meaned rank weight: rank - (N+1)/2
    demeaned_ranks = ranks - (N_t + 1) / 2
    
    # Separate positive and negative weights
    pos_weights = demeaned_ranks[demeaned_ranks > 0]
    neg_weights = demeaned_ranks[demeaned_ranks < 0]
    
    # Scale positive weights to sum to 1 (if any)
    weights = demeaned_ranks.copy()
    if len(pos_weights) > 0:
        pos_scale = 1 / pos_weights.sum()
        weights[weights > 0] = weights[weights > 0] * pos_scale
    
    # Scale negative weights to sum to -1 (if any)
    if len(neg_weights) > 0:
        neg_scale = 1 / neg_weights.sum()
        weights[weights < 0] = weights[weights < 0] * neg_scale * -1
    
    return weights, ranks, demeaned_ranks

def main():
    # Using the imported tickers list from the import statement
    # from tickers import tickers
    
    # Fetch 18 months of data (giving extra buffer)
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(months=18)).strftime('%Y-%m-%d')
    window = 252  # Approx 12 months of trading days
    
    print(f"Fetching data from {start} to today for {len(tickers)} tickers")
    
    # Initialize a list to store individual ticker data Series
    ticker_data_list = []
    valid_tickers = []
    
    # Download data for each ticker individually to avoid rate limits
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"Downloading {i}/{len(tickers)}: {ticker}")
            ticker_data = yf.download(ticker, start=start, progress=False, auto_adjust=False)
            
            if not ticker_data.empty:
                # Extract the Adjusted Close prices and save as a Series
                adj_close = ticker_data['Adj Close']
                ticker_data_list.append(adj_close)
                valid_tickers.append(ticker)
            else:
                print(f"Warning: No data available for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    # Combine all ticker data at once using pd.concat
    if ticker_data_list:
        data = pd.concat(ticker_data_list, axis=1)
        data.columns = valid_tickers
    else:
        data = pd.DataFrame()
    
    print(f"Data shape: {data.shape}, Date range: {data.index.min()} to {data.index.max()}")
    
    # Check for missing data
    missing_data = data.isna().sum()
    if missing_data.sum() > 0:
        print(f"Missing data points per ticker: {missing_data[missing_data > 0]}")
    
    skewness_df = pd.DataFrame(index=data.index)
    
    for ticker in tickers:
        if ticker in data.columns:
            skewness_df[ticker] = compute_rolling_skewness(data[ticker], window=window)
            # Print how many valid skewness values we have
            valid_count = skewness_df[ticker].notna().sum()
            print(f"{ticker}: {valid_count} valid skewness values out of {len(skewness_df)}")
        else:
            print(f"Warning: No data for {ticker}")
    
    # Continue only if we have valid data
    if skewness_df.dropna(how='all').empty:
        print("ERROR: No valid skewness values were calculated!")
        return
    
    # Get last available skewness values
    last_date = skewness_df.dropna(how='all').tail(1).index[0]
    last_skew = skewness_df.loc[last_date]
    
    if last_skew.dropna().empty:
        print("ERROR: No valid skewness values for the last date!")
        return
    
    # Calculate portfolio weights based on negative skewness ranking
    weights, ranks, demeaned_ranks = calculate_portfolio_weights(last_skew.dropna())
    
    # Create a result DataFrame with skewness, ranks, and weights
    result_df = pd.DataFrame({
        'skewness': last_skew[weights.index],
        'rank': ranks,
        'demeaned_rank': demeaned_ranks,
        'weight': weights
    }).sort_values('weight', ascending=False)  # Sort by weight
    
    # Output to CSV files
    output_file = 'skewness.csv'
    result_df.to_csv(output_file)
    
    print(f"\n=== Portfolio Construction Results ===")
    print(f"Date: {last_date}")
    print(f"\nOutput saved to: {output_file}")
    print("\nSkewness Values, Ranks, and Portfolio Weights:")
    print(result_df)
    
    # Show summary of weights
    print("\nWeight Summary:")
    print(f"Sum of absolute weights: {weights.abs().sum():.4f}")
    print(f"Sum of positive weights: {weights[weights > 0].sum():.4f}")
    print(f"Sum of negative weights: {weights[weights < 0].sum():.4f}")
    
    # Show most attractive stocks (most negative skewness, highest weight)
    print("\nMost Attractive Stocks (highest positive weights):")
    print(result_df.sort_values('weight', ascending=False))

if __name__ == "__main__":
    main()