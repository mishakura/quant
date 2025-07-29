import yfinance as yf
import numpy as np
import pandas as pd

# Define tickers for Bitcoin and Ethereum
TICKERS = ['BTC-USD', 'ETH-USD']

# Download daily price data
prices = yf.download(TICKERS, start='2020-01-01')['Close']

# Show when the data starts for each asset
for ticker in TICKERS:
    if ticker in prices and not prices[ticker].dropna().empty:
        first_date = prices[ticker].dropna().index[0]
        print(f"{ticker} data starts: {first_date.date()}")
    else:
        print(f"No data found for {ticker}.")

# Calculate daily returns

# Calculate daily returns and log returns
returns = prices.pct_change().dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()

# Compute annualized expected return and standard deviation

# Annualize using 365 days for crypto

# Arithmetic mean annualized return
annualized_return = returns.mean() * 365
annualized_std = returns.std() * np.sqrt(365)



print("\nAnnualized Expected Return (Arithmetic):")
print(annualized_return)
print("\nAnnualized Standard Deviation:")
print(annualized_std)
