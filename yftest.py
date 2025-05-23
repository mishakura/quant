import yfinance as yf

t = yf.Ticker("SPOT")
info = t.info
print(info)
 

