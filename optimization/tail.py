import yfinance as yf
import numpy as np
import powerlaw
from hurst import compute_Hc
import pandas as pd
import matplotlib.pyplot as plt

# Define list of tickers
tickers = ["SPY", "IJH", "IWM", "EEM", "VEA", "GLD", "SLV","USO","BTC-USD"]

# Dictionary to store results
results = {}

# Process each ticker
for ticker in tickers:
    print(f"\nAnalyzing {ticker}...")
    try:
        # Download data
        data = yf.download(ticker, start="1930-01-01", auto_adjust=True)['Close']
        returns = np.log(data / data.shift(1)).dropna()
        
        # Hurst Exponent
        H, c, data_reg = compute_Hc(data.values, kind='price')
        
        # Tail Index (alpha)
        returns_abs = np.array(returns.abs()).flatten()
        returns_abs = returns_abs[returns_abs > 0]
        
        fit = powerlaw.Fit(returns_abs, xmin=0.01)
        alpha = fit.alpha
        
        # Distribution comparison
        R, p = fit.distribution_compare('power_law', 'lognormal')
        
        # Store results
        results[ticker] = {
            'alpha': alpha,
            'hurst': H,
            'R': R,
            'p': p
        }
        
        print(f"  Hurst exponent: {H:.2f}")
        print(f"  Tail index (alpha): {alpha:.2f}")
        print(f"  R: {R:.4f}, p: {p:.4e}")
        
    except Exception as e:
        print(f"Error with {ticker}: {str(e)}")

# Create weights based on tail index and Hurst exponent
if results:
    # Create DataFrame with results
    df_results = pd.DataFrame.from_dict(results, orient='index')
    
    # Calculate trending factor (assets with H > 0.5 are trending)
    df_results['trend_factor'] = np.maximum(0, df_results['hurst'] - 0.5)  # How much above 0.5
    
    # Combined weight: alpha^2 * (1 + trend_factor)
    # Square the tail index to make its impact more pronounced
    df_results['weight'] = (df_results['alpha'] ** 2) * (1 + df_results['trend_factor'] * 2)
    
    # Normalize weights to sum to 1
    df_results['weight'] = df_results['weight'] / df_results['weight'].sum()
    
    # Sort by weight (descending)
    df_results = df_results.sort_values('weight', ascending=False)
    
    # Display results
    print("\n--- Portfolio Weights (Combined Tail + Trend) ---")
    for ticker, row in df_results.iterrows():
        print(f"{ticker}: {row['weight']:.4f} (alpha: {row['alpha']:.2f}, hurst: {row['hurst']:.2f})")
    
    # Plot weights
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_results.index, df_results['weight'])
    plt.xlabel('Ticker')
    plt.ylabel('Portfolio Weight')
    plt.title('Portfolio Weights Based on Tail Index and Hurst Exponent')
    plt.xticks(rotation=45)
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # Create a scatter plot to visualize the relationship
    plt.figure(figsize=(10, 6))
    plt.scatter(df_results['hurst'], df_results['alpha'], s=100)
    
    # Add ticker labels to each point
    for i, ticker in enumerate(df_results.index):
        plt.annotate(ticker, 
                    (df_results['hurst'].iloc[i], df_results['alpha'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Hurst Exponent')
    plt.ylabel('Tail Index (alpha)')
    plt.title('Relationship between Hurst Exponent and Tail Index')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

