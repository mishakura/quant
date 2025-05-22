import yfinance as yf
ticker = yf.Ticker("AAPL")
balance_quarterly = ticker.get_balance_sheet(freq="quarterly")

print(balance_quarterly.columns)