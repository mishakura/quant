"""Hello, we are going to code a momentum strategy (long only). We are going to use the algorithm from the book Quantitative Momentum from Wesley R Gray and Jack R Vogel.

- We are going to use yahoo finance data
- The universe of assets its in tickers.py (its a tickers array)
_ We calculate the generic momentum measure as the total return (including dividends) of a stock over some particular look-back
period (e.g., the past 12 months) and skip the most recent month. We calculate this measure for all stocks in our investment universe.
- Next we will filter does stock that had a recent and really large short-term spike (becouse those stocks usually presents mean reversion characteristics). The calculation
for the measure is described as follows:
FIP = 1∗[% negative − % positive]

In step 4, we determine the model portfolio and
conduct our rebalance at the end of February, May, August, and November
to exploit seasonality effects.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from tickers import tickers




today = datetime.today()
end_date = today - pd.DateOffset(months=1)
start_date = (end_date - pd.DateOffset(months=12))

us_exchanges = {'NMS', 'NYQ', 'AMEX', 'BATS', 'ARCX', 'PCX', 'NGM', 'NSC', 'XASE', 'XNAS', 'XNYS'}
force_us_tickers = {'GGAL', 'RGTI', 'NG', 'RIOT', 'SATL', 'SDA'}  # Always treat these as US

momentum_scores = {}
skipped_tickers = []

# Step 1: Calculate momentum for all stocks
for ticker in tickers:
    print(f"Checking exchange for {ticker}...")
    t = yf.Ticker(ticker)
    try:
        info = t.info
        exchange = info.get('exchange', '').upper()
    except Exception as e:
        print(f"Skipping {ticker} due to error fetching info: {e}")
        skipped_tickers.append((ticker, f"info error: {e}"))
        continue

    # Pass if in force_us_tickers, otherwise check exchange
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
    if len(monthly_returns) < 2:
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
top_momentum_df = momentum_df.head(top_10pct_count).copy()  # Add .copy() here

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









