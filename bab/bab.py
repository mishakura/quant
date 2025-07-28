import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'UNH', 'V']  # Example tickers
market = '^GSPC'  # S&P 500 as market proxy
today = datetime.today()
one_year_ago = today - timedelta(days=365)
five_years_ago = today - timedelta(days=5*365)

# Download data
raw_data = yf.download(tickers + [market], start=five_years_ago.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
data = raw_data.xs('Adj Close', axis=1, level=1)

# Calculate daily log returns (for volatility, 1 year)
daily_returns = np.log(data / data.shift(1))
daily_returns_1y = daily_returns.loc[one_year_ago:]

# Calculate 3-day log returns (for correlation, 5 years)
returns_3d = np.log(data / data.shift(3))
returns_3d_5y = returns_3d.loc[five_years_ago:]

# Volatility (std of daily returns over 1 year)
volatility = daily_returns_1y[tickers].std()

# Correlation (3-day returns over 5 years, shrinkage towards 0)
corr_matrix = returns_3d_5y[tickers + [market]].corr()
shrinkage = 0.1  # Shrink correlations 10% towards zero
corr_with_market = corr_matrix.loc[tickers, market] * (1 - shrinkage)

# Beta = corr(stock, market) * (vol(stock)/vol(market))
vol_market = daily_returns_1y[market].std()
beta = corr_with_market * (volatility / vol_market)

# Rank stocks by beta (low to high)
ranking = beta.sort_values().reset_index()
ranking.columns = ['Ticker', 'Beta']
ranking['Rank'] = ranking['Beta'].rank(method='min')

# Output to Excel
ranking.to_excel('bab_beta_ranking.xlsx', index=False)
print("Excel file 'bab_beta_ranking.xlsx' created with beta rankings.")
