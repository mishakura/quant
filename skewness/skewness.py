import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from tickers import tickers

##Now it ask for at least 80% of data to compute the 252 window
def compute_rolling_skewness(prices, window=252):
    returns = prices.pct_change(fill_method=None)
    min_periods = int(0.8 * window)
    rolling_skew = returns.rolling(window=window, min_periods=min_periods).apply(
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

def is_etf(ticker):
    """Check if a ticker is an ETF using yfinance info."""
    try:
        info = yf.Ticker(ticker).info
        return info.get('quoteType', '').lower() == 'etf'
    except Exception:
        return False

def get_market_cap_category(market_cap):
    """Classify market cap."""
    if market_cap is None or np.isnan(market_cap):
        return 'unknown'
    if market_cap >= 10_000_000_000:
        return 'large'
    elif market_cap >= 2_000_000_000:
        return 'medium'
    else:
        return 'small'

def main():
    # Using the imported tickers list from the import statement
    # from tickers import tickers
    
    # Fix the date range - use current date as end date
    from datetime import datetime, timedelta
    
    end = '2025-09-01'  # Today's date
    start = '2022-01-01'  # Go back further to ensure enough history
    window = 252  # Approx 12 months of trading days
    
    print(f"Fetching data from {start} to {end} for {len(tickers)} tickers")
    
    # Initialize a list to store individual ticker data Series
    ticker_data_list = []
    valid_tickers = []
    
    # Download data for each ticker individually to avoid rate limits
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"Downloading {i}/{len(tickers)}: {ticker}")
            ticker_data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            
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
    
    # Drop dates with more than 20% missing data
    data = data.dropna(thresh=int(0.8 * len(data.columns)))
    
    # Instead of assigning columns in a loop, build a dict:
    skewness_dict = {}
    for ticker in tickers:
        if ticker in data.columns:
            skewness_series = compute_rolling_skewness(data[ticker], window=window)
            skewness_dict[ticker] = skewness_series
            valid_count = skewness_series.notna().sum()
        else:
            print(f"Warning: No data for {ticker}")

    # Combine all at once to avoid fragmentation
    if skewness_dict:
        skewness_df = pd.DataFrame(skewness_dict)
    else:
        skewness_df = pd.DataFrame(index=data.index)
    
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
    
    # Filter tickers: keep ETFs, and only medium/large cap stocks
    filtered_tickers = []
    for ticker in valid_tickers:
        try:
            info = yf.Ticker(ticker).info
            if is_etf(ticker):
                filtered_tickers.append(ticker)
            else:
                market_cap = info.get('marketCap', np.nan)
                cap_cat = get_market_cap_category(market_cap)
                if cap_cat in ['medium', 'large']:
                    filtered_tickers.append(ticker)
                else:
                    print(f"Excluding {ticker}: market cap category {cap_cat}")
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")

    # Filter data and skewness_dict to only include filtered_tickers
    data = data[filtered_tickers]
    skewness_dict = {k: v for k, v in skewness_dict.items() if k in filtered_tickers}

    # Combine all at once to avoid fragmentation
    if skewness_dict:
        skewness_df = pd.DataFrame(skewness_dict)
    else:
        skewness_df = pd.DataFrame(index=data.index)
    
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
    
    # Output top 10% by positive rank
    top_10p = result_df[result_df['rank'] <= np.percentile(result_df['rank'], 10)]
    top_10p_file = 'skewness_top10p.csv'
    top_10p.to_csv(top_10p_file)
    print(f"\nTop 10% by positive rank saved to: {top_10p_file}")

if __name__ == "__main__":
    main()