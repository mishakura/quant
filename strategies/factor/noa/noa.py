import yfinance as yf
import pandas as pd

def get_noa(ticker):
    t = yf.Ticker(ticker)
    bs = t.balance_sheet

    if bs.empty:
        print(f"No balance sheet data for {ticker}")
        return None

    print(f"\nBalance sheet index for {ticker}:")
    print(bs.index.tolist())
    print("\nBalance sheet data:")
    print(bs)

    latest = bs.columns[0]

    def safe_get(label):
        return bs.loc[label, latest] if label in bs.index else 0

    # Gather stats using your provided keys
    total_assets = safe_get('Total Assets')
    cash_and_st_inv = safe_get('Cash Cash Equivalents And Short Term Investments')
    total_liab = safe_get('Total Liabilities Net Minority Interest')
    total_debt = safe_get('Total Debt')

    print(f"Stats for {ticker} (period: {latest}):")
    print(f"  Total Assets:         {total_assets:,.2f}")
    print(f"  Cash & ST Inv:        {cash_and_st_inv:,.2f}")
    print(f"  Total Liabilities:    {total_liab:,.2f}")
    print(f"  Total Debt:           {total_debt:,.2f}")

    operating_assets = total_assets - cash_and_st_inv
    operating_liab = total_liab - total_debt
    noa = operating_assets - operating_liab

    print(f"  Operating Assets:     {operating_assets:,.2f}")
    print(f"  Operating Liabilities:{operating_liab:,.2f}")
    print(f"  Net Operating Assets: {noa:,.2f}\n")

    return noa

if __name__ == "__main__":
    ticker = "AAPL"  # Change to your ticker
    noa = get_noa(ticker)
    if noa is not None:
        print(f"Net Operating Assets (NOA) for {ticker}: {noa:,.2f}")
    else:
        print(f"Could not calculate NOA for {ticker}")