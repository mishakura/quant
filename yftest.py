import yfinance as yf
from datetime import datetime, timedelta

# Define the ticker
ticker = yf.Ticker("AAPL")

# Get all available daily historical data
data = ticker.history(period="max", interval="1d")

# Print the last 5 rows with their index (dates)
print(data.head())