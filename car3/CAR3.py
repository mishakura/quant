import yfinance as yf
from datetime import datetime, timedelta
import pytz
import pandas as pd

# Define the tickers
from tickers import tickers 

results = []

us_exchanges = {'NMS', 'NYQ', 'AMEX','NGM','NCM'}

for ticker_symbol in tickers:
    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
        exchange = info.get('exchange', '').upper()
        market_cap = info.get('marketCap', 0)
    except Exception as e:
        print(f"Skipping {ticker_symbol}: error fetching info ({e})")
        continue
    if exchange not in us_exchanges:
        print(f"Skipping {ticker_symbol}: not a US exchange ({exchange})")
        continue
    if not market_cap or market_cap < 2_000_000_000:
        print(f"Skipping {ticker_symbol}: market cap too small ({market_cap})")
        continue

    print(f"Downloading data for {ticker_symbol}...")  # <-- Add this line

    earnings_history = ticker.earnings_dates

    if earnings_history is not None and not earnings_history.empty:
        nyse_tz = pytz.timezone('America/New_York')
        current_time = datetime.now(nyse_tz)
        past_earnings = earnings_history[earnings_history.index < current_time]
        if past_earnings.empty:
            continue
        latest_earnings = past_earnings.iloc[0]
        earnings_date = latest_earnings.name

        start = (earnings_date - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
        end = (earnings_date + pd.Timedelta(days=20)).strftime('%Y-%m-%d')
        price_data = ticker.history(start=start, end=end)

        trading_days = price_data.index
        earnings_idx = trading_days.get_indexer([earnings_date], method='nearest')[0]
        window_indices = [earnings_idx - 1, earnings_idx, earnings_idx + 1]
        window_indices = [i for i in window_indices if 0 <= i < len(trading_days)]
        window_dates = trading_days[window_indices]
        window_data = price_data.loc[window_dates, 'Close']

        if len(window_data) < 2:
            continue

        # Get starting and ending dates/prices
        starting_date = window_dates[0].strftime('%Y-%m-%d')
        ending_date = window_dates[-1].strftime('%Y-%m-%d')
        starting_price = window_data.iloc[0]
        ending_price = window_data.iloc[-1]

        cumulative_return = (ending_price / starting_price) - 1

        spy = yf.Ticker("SPY")
        spy_price_data = spy.history(start=start, end=end)
        spy_window_data = spy_price_data.loc[window_dates, 'Close']
        if len(spy_window_data) < 2:
            continue
        spy_cumulative_return = (spy_window_data.iloc[-1] / spy_window_data.iloc[0]) - 1

        excess_return = cumulative_return - spy_cumulative_return

        results.append({
            "Ticker": ticker_symbol,
            "Starting Date": starting_date,
            "Ending Date": ending_date,
            "Starting Price": starting_price,
            "Ending Price": ending_price,
            "Cumulative Return": cumulative_return,
            "SPY Cumulative Return": spy_cumulative_return,
            "Excess Return": excess_return
        })

# Output to CSV
df = pd.DataFrame(results)
df.to_csv("CAR3.csv", index=False)
print("Results saved to CAR3.csv")

# Take top 10% by Excess Return and output to another CSV
if not df.empty:
    top_10pct_count = max(1, int(len(df) * 0.1))
    top_excess_df = df.sort_values(by="Excess Return", ascending=False).head(top_10pct_count)
    top_excess_df.to_csv("CAR3_top10pct.csv", index=False)
    print("Top 10% by Excess Return saved to CAR3_top10pct.csv")
