import yfinance as yf

ticker = yf.Ticker("NFLX")

quarterly = ticker.get_income_stmt(freq="quarterly")
print(len(quarterly.columns))
