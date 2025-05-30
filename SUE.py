import yfinance as yf
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

tickers = ["msft", "aapl", "nflx"]
results = []

for ticker_symbol in tickers:
    ticker = yf.Ticker(ticker_symbol)

    # Get yearly income statements only
    yearly_income = ticker.get_income_stmt(freq="yearly")

    # Initialize result dictionary
    result = {"Ticker": ticker_symbol, "Last EPS YoY Change (%)": None, "EPS YoY Stdev (annual)": None}

    try:
        # Get yearly EPS and sort by date
        eps_yearly = yearly_income.loc["BasicEPS"].sort_index().astype(float)
        eps_yearly.index = pd.to_datetime(eps_yearly.index)
        
        # Calculate YoY change
        yoy_change = eps_yearly.pct_change()
        
        # Get the most recent YoY change
        last_yoy_change = yoy_change.dropna().iloc[-1] if not yoy_change.empty else None
        result["Last EPS YoY Change (%)"] = last_yoy_change
        
        # Calculate standard deviation of YoY changes
        stdev_eps_change = yoy_change.std()
        result["EPS YoY Stdev (annual)"] = stdev_eps_change
        
        print(f"\n{ticker_symbol.upper()} Analysis:")
        print("Yearly EPS values:")
        print(eps_yearly.tail())
        print("\nYear-over-Year % changes:")
        print(yoy_change.tail())
        print(f"EPS YoY Stdev (annual): {stdev_eps_change:.4f}\n")
        
    except Exception as e:
        print(f"{ticker_symbol} - Error in calculations: {e}")

    results.append(result)

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("SUE_results.csv", index=False)