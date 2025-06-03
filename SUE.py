import requests
import numpy as np
import csv
import time
from tickers import tickers

API_KEY = "zYbh0y_1wnVdlzSjCFVk6dHHMAMbOEm5"
TICKERS = tickers
URL = f"https://api.polygon.io/vX/reference/financials"

results = []
not_found = []

for TICKER in TICKERS:
    print(f"Downloading data for {TICKER}...")
    params = {
        "ticker": TICKER,
        "limit": 50,
        "type": "Q",
        "apiKey": API_KEY
    }
    try:
        response = requests.get(URL, params=params)
        print(f"Status code for {TICKER}: {response.status_code}")
        data = response.json()
        time.sleep(20)
    except Exception as e:
        print(f"Error downloading data for {TICKER}: {e}")
        not_found.append(TICKER)
        time.sleep(20)
        continue

    eps_by_quarter = []
    if "results" in data and data["results"]:
        for report in data["results"]:
            fiscal_period = report.get("fiscal_period", "N/A")
            fiscal_year = report.get("fiscal_year", "N/A")
            if fiscal_period in ["Q1", "Q2", "Q3", "Q4"]:
                income_statement = report.get("financials", {}).get("income_statement", {})
                eps = income_statement.get("basic_earnings_per_share", {}).get("value", None)
                if eps is not None:
                    eps_by_quarter.append((int(fiscal_year), fiscal_period, eps))
    else:
        not_found.append(TICKER)
        continue

    if len(eps_by_quarter) < 9:
        not_found.append(TICKER)
        continue

    quarter_order = {"Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}
    eps_by_quarter.sort(key=lambda x: (x[0], quarter_order[x[1]]), reverse=True)
    eps_dict = {(y, q): eps for y, q, eps in eps_by_quarter}

    # Calculate YoY % changes for all available quarters
    yoy_changes_full = []
    for y, q, eps in eps_by_quarter:
        prev_year = y - 1
        prev_eps = eps_dict.get((prev_year, q))
        yoy_pct_change = None
        if prev_eps is not None and prev_eps != 0:
            yoy_pct_change = (eps - prev_eps) / abs(prev_eps)
            yoy_changes_full.append((y, q, yoy_pct_change))

    # Calculate most recent SUE (using the most recent YoY change and previous 8)
    quarter_order = {"Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}
    yoy_changes_full_sorted = sorted(yoy_changes_full, key=lambda x: (x[0], quarter_order[x[1]]), reverse=True)
    if len(yoy_changes_full_sorted) >= 9:
        most_recent_change = yoy_changes_full_sorted[0][2]
        prev_8_changes = [chg[2] for chg in yoy_changes_full_sorted[1:9]]
        if all(x is not None for x in prev_8_changes):
            stdev = np.std(prev_8_changes, ddof=1)
            sue = most_recent_change / stdev if stdev != 0 else float('inf')
            results.append([TICKER, sue])
        else:
            not_found.append(TICKER)
    else:
        not_found.append(TICKER)


# Write SUE values to CSV (most recent SUE per ticker)
with open("sue_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Ticker", "SUE"])
    writer.writerows(results)

print("CSV output written to sue_output.csv")
print("Tickers with no data:", not_found)