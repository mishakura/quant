import yfinance as yf

# Fetch VXX data from 1930-01-01 to current date
vxx = yf.Ticker("^VIX")
data = vxx.history(start="1930-01-01")
print(data)