

import yfinance as yf
import pandas as pd
from datetime import datetime
from tickers import tickers

""""
Yahoo finance for the end date takes your date - 1 day
So its important to set the end date to rebalance date + 1
The rebalance is on the end of Feb, May, Aug , Nov.
So the correct dates to input are:
2025-03-01
2025-06-01
2025-09-01
2025-12-01
"""

user_end_date = pd.Timestamp("2025-09-01")  # INPUT THE DATES LISTED ABOVE THE CODE

# 1. Check if input date is in the future
today = pd.Timestamp(datetime.today().date())
if user_end_date > today:
    print(f"ERROR: The input date {user_end_date.date()} is in the future. Please use a valid date up to {today.date()}.")
    exit()

# Use user input as end_date
end_date = user_end_date

start_date = end_date - pd.DateOffset(months=13)
def get_rebalance_dates(years):
    if isinstance(years, int):
        years = [years]
    months = [2, 5, 8, 11]  # Feb, May, Aug, Nov
    dates = []
    for year in years:
        for month in months:
            last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            rebalance_day = last_day + pd.Timedelta(days=1)
            dates.append(rebalance_day)
    return dates

us_exchanges = {'NMS', 'NYQ', 'AMEX', 'BATS', 'ARCX', 'PCX', 'NGM', 'NSC', 'XASE', 'XNAS', 'XNYS'}
force_us_tickers = {'GGAL', 'RGTI', 'NG', 'RIOT', 'SATL', 'SDA'}  # Always treat these as US

momentum_scores = {}
skipped_tickers = []

# Step 1: Calculate momentum for all stocks
for ticker in tickers:
    t = yf.Ticker(ticker)
    try:
        info = t.info
        exchange = info.get('exchange', '').upper()
    except Exception as e:
        print(f"Skipping {ticker} due to error fetching info: {e}")
        skipped_tickers.append((ticker, f"info error: {e}"))
        continue

    if ticker not in force_us_tickers and exchange not in us_exchanges:
        print(f"Skipping {ticker} (exchange: {exchange})")
        skipped_tickers.append((ticker, f"exchange: {exchange}"))
        continue

    print(f"Downloading data for {ticker}...")
    data = t.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
    if data.empty or "Close" not in data:
        skipped_tickers.append((ticker, "no data"))
        continue
    adj_close = data["Close"]
    adj_close.index = pd.to_datetime(adj_close.index)
    monthly_prices = adj_close.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    monthly_returns = monthly_returns[-12:]
    if len(monthly_returns) < 12:
        skipped_tickers.append((ticker, "not enough monthly returns"))
        continue
    momentum = (monthly_returns[:-1] + 1).prod() - 1
    if pd.isna(momentum):
        skipped_tickers.append((ticker, "momentum is NaN"))
        continue
    momentum_scores[ticker] = momentum

# Step 2: Pick top 10% by momentum
if len(momentum_scores) == 0:
    print("No stocks with positive momentum found.")
    exit()

momentum_df = pd.DataFrame(list(momentum_scores.items()), columns=['Ticker', 'Momentum'])
momentum_df = momentum_df.sort_values(by='Momentum', ascending=False)
top_10pct_count = max(1, int(len(momentum_df) * 0.10))
top_momentum_df = momentum_df.head(top_10pct_count).copy()

# Step 3: Calculate FIP for top 10% momentum stocks
fip_scores = {}
for ticker in top_momentum_df['Ticker']:
    t = yf.Ticker(ticker)
    data = t.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
    if data.empty or "Close" not in data:
        fip_scores[ticker] = None
        continue
    adj_close = data["Close"]
    adj_close.index = pd.to_datetime(adj_close.index)
    daily_returns = adj_close.pct_change().dropna()[-252:]
    num_days = len(daily_returns)
    if num_days > 0:
        pct_negative = (daily_returns < 0).sum() / num_days
        pct_positive = (daily_returns > 0).sum() / num_days
        sign_momentum = 1  # Always 1 since momentum > 0
        fip = sign_momentum * (pct_negative - pct_positive)
    else:
        fip = None
    fip_scores[ticker] = fip

top_momentum_df['FIP'] = top_momentum_df['Ticker'].map(fip_scores)

# Step 4: Pick best 50% (lowest) FIP
top_momentum_df = top_momentum_df.dropna(subset=['FIP'])
top_momentum_df = top_momentum_df.sort_values(by='FIP')
top_50pct_count = max(1, int(len(top_momentum_df) * 0.5))
final_df = top_momentum_df.head(top_50pct_count)

# Step 5: Save to CSV
final_df.to_csv('momentum_fip_scores.csv', index=False)
print("Momentum and FIP scores saved to momentum_fip_scores.csv")

# Example usage:
rebalance_dates = get_rebalance_dates([2025])









