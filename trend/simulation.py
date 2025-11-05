import math
from pathlib import Path
import pandas as pd

# Simulation of position sizing and PnL using trades/master.csv

TRADES_CSV = Path(__file__).parent.joinpath("trades", "master.csv")
# write results to the trend folder (same folder as this script)
OUTPUT_CSV = Path(__file__).parent.joinpath("simulation_results.csv")

START_CAPITAL = 100_000.0
RISK = 0.0005  

def load_trades(path: Path) -> pd.DataFrame:
    # read normally then explicitly convert Date to datetime to avoid dtype=str overriding parse_dates
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # ensure numeric columns converted
    for col in ("High", "Low", "Close", "Max200", "Min200", "Max100", "Min100",
                "ATR25", "EntryPrice", "StopPrice", "ExitPrice", "PnL_Percent"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # sort by date then ticker alphabetically to respect chronological + alpha ordering
    if "Ticker" in df.columns:
        df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    else:
        df = df.sort_values(["Date"]).reset_index(drop=True)
    return df

def simulate(df: pd.DataFrame, start_capital: float, risk: float):
    capital = float(start_capital)
    positions = {}  # ticker -> position dict
    results = []

    def _fmt_date(d):
        if pd.isna(d):
            return ""
        return pd.to_datetime(d).strftime("%Y-%m-%d")
    
    for i, row in df.iterrows():
        ticker = row.get("Ticker")
        entry_price = row.get("EntryPrice")
        exit_price = row.get("ExitPrice")
        stop_price = row.get("StopPrice")
        atr = row.get("ATR25")
        date = row.get("Date")

        # ENTRY: presence of an EntryPrice and no open position for this ticker
        if pd.notna(entry_price) and ticker not in positions:
            if pd.isna(atr) or atr <= 0:
                # skip if ATR invalid
                continue
            # number of contracts (integer)
            contracts = int((capital * risk) / atr)
            if contracts <= 0:
                # cannot size a position with current capital/ATR
                continue
            usd_position = contracts * entry_price
            positions[ticker] = {
                "entry_date": date,
                "entry_price": float(entry_price),
                "stop_price": float(stop_price) if pd.notna(stop_price) else None,
                "atr_on_entry": float(atr),
                "contracts": contracts,
                "usd_position": usd_position,
                "capital_at_entry": capital,
            }

        # EXIT: presence of ExitPrice and there is an open position for this ticker
        if pd.notna(exit_price) and ticker in positions:
            pos = positions.pop(ticker)
            entry_price_f = pos["entry_price"]
            contracts = pos["contracts"]
            usd_position = pos["usd_position"]
            capital_before = capital

            # PnL per contract (assumes long trades)
            pnl_per_contract = float(exit_price) - entry_price_f
            pnl_usd = pnl_per_contract * contracts

            # update capital
            capital = capital + pnl_usd

            # percent returns
            pnl_pct_on_position = (pnl_usd / usd_position * 100) if usd_position != 0 else None
            pnl_pct_on_capital = (pnl_usd / pos["capital_at_entry"] * 100) if pos["capital_at_entry"] != 0 else None

            results.append({
                "Ticker": ticker,
                "EntryDate": _fmt_date(pos["entry_date"]),
                "ExitDate": _fmt_date(date),
                "EntryPrice": entry_price_f,
                "ExitPrice": float(exit_price),
                "Contracts": contracts,
                "USD_Position": usd_position,
                "PnL_USD": pnl_usd,
                "PnL_pct_on_position": pnl_pct_on_position,
                "PnL_pct_on_capital": pnl_pct_on_capital,
                "Capital_before": capital_before,
                "Capital_after": capital,
                "ATR_on_entry": pos["atr_on_entry"],
                "StopPrice": pos["stop_price"],
                "Reason": row.get("Reason", ""),
                "Signal": row.get("Signal", ""),
            })

    # if any positions remain open at file end, optionally close them at last available price (not done here)
    return start_capital, capital, results

def main():
    if not TRADES_CSV.exists():
        print(f"trades file not found: {TRADES_CSV}")
        return

    df = load_trades(TRADES_CSV)
    start, final_capital, trades = simulate(df, START_CAPITAL, RISK)

    out_df = pd.DataFrame(trades)
    out_df.to_csv(OUTPUT_CSV, index=False)

    wins = sum(1 for t in trades if t["PnL_USD"] > 0)
    losses = sum(1 for t in trades if t["PnL_USD"] <= 0)
    total_trades = len(trades)
    total_pnl = sum(t["PnL_USD"] for t in trades)

    print(f"Start capital: {start:,.2f}")
    print(f"End capital:   {final_capital:,.2f}")
    print(f"Total trades:  {total_trades}, Wins: {wins}, Losses: {losses}")
    print(f"Total PnL USD: {total_pnl:,.2f}")
    print(f"Results exported to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()