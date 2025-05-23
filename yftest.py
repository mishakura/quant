import pandas as pd
import yfinance as yf

ticker = yf.Ticker("MO")
info = balance_annual = ticker.get_balance_sheet(freq="yearly")

# Print all available keys (columns) in the info dictionary
#print(list(info.keys()))
print(info)