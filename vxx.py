import yfinance as yf

# Fetch NVDA data from 2015-01-01 to today (2025-11-26)
nvda = yf.Ticker("NVDA")
data = nvda.history(start="2015-01-01", end="2025-11-26")

# Calculate total return using adjusted close prices (accounts for dividends and splits)
initial_price = data['Close'].iloc[0]  # First available close price in 2015
final_price = data['Close'].iloc[-1]    # Latest close price
total_return = (final_price / initial_price) - 1

print(f"First Day Price (2015-01-02): ${initial_price:.2f}")
print(f"Last Day Price (2025-11-25): ${final_price:.2f}")
print(f"Total Return for NVDA from 2015 to 2025-11-26: {total_return:.2%}")