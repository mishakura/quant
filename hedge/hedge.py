import yfinance as yf

# Fetch historical 1-month VIX futures data
vix_futures_data = yf.download("^VFTW1", start="2020-01-01", end="2025-11-12")
print(vix_futures_data.head())
##https://www.cboe.com/us/futures/market_statistics/historical_data/