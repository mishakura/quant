import yfinance as yf

# Fetch NVDA data from the earliest available date to today (2025-11-26)
nvda = yf.Ticker("SH")
data = nvda.history(start="1900-01-01", end="2025-12-01")

# Calculate total return using adjusted close prices (accounts for dividends and splits)
initial_price = data['Close'].iloc[0]  # First available close price
final_price = data['Close'].iloc[-1]    # Latest close price
total_return = (final_price / initial_price) - 1

print(f"First Day Price ({data.index[0].date()}): ${initial_price:.2f}")
print(f"Last Day Price ({data.index[-1].date()}): ${final_price:.2f}")
print(f"Total Return for NVDA from {data.index[0].date()} to 2025-11-26: {total_return:.2%}")