import yfinance as yf
import pandas as pd

# Download all available historical data for ARGT
ticker = "ARGT"
data = yf.download(ticker, progress=False)

# Use 'Close' for return calculations (auto_adjust=True by default)
prices = data['Close']

# Print the start date of the data
start_date = prices.index.min()
print(f"Data starts on: {start_date.date()}")

# Calculate daily returns
daily_returns = prices.pct_change().dropna()

# Calculate mean annual return and mean annual standard deviation
mean_annual_return = daily_returns.mean() * 252
mean_annual_stdev = daily_returns.std() * (252 ** 0.5)

# If mean_annual_return is a Series, get the value
if isinstance(mean_annual_return, pd.Series):
    mean_annual_return = mean_annual_return.values[0]
    mean_annual_stdev = mean_annual_stdev.values[0]

print(f"Mean annual return: {mean_annual_return:.4f}")
print(f"Mean annual standard deviation: {mean_annual_stdev:.4f}")