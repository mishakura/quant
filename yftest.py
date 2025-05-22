import pandas as pd
import yfinance as yf

ticker = yf.Ticker("AVGO")
info = ticker.info

# Print all available keys (columns) in the info dictionary
#print(list(info.keys()))
print(info)