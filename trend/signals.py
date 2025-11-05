import os
from pathlib import Path
import pandas as pd
import numpy as np

## Calculate signals.

# Directory containing the CSVs (adjust if different)
INDICATORS_DIR = Path(os.path.join(os.path.dirname(__file__), "indicators"))

def process_file(csv_path: Path, inplace: bool = True):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure required columns exist
    required = {"Date", "High", "Low", "Close", "Max200", "Min200", "Max100", "Min100", "ATR25"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Prepare output columns
    df["Signal"] = 0  # 1 if long on that day, 0 otherwise
    df["Reason"] = ""  # short reason text
    # New debug columns: maintain until trade closed
    df["EntryPrice"] = np.nan
    df["StopPrice"] = np.nan
    df["ExitPrice"] = np.nan
    df["PnL_Percent"] = np.nan

    in_trade = False
    entry_price = None
    entry_atr = None

    # iterate rows; start at 0 but we need previous-day values so handle index 0 separately
    for i in range(len(df)):
        if i == 0:
            df.at[i, "Signal"] = 0
            df.at[i, "Reason"] = "no_prev"
            # EntryPrice / StopPrice / ExitPrice / PnL_Percent remain NaN
            continue

        today_close = float(df.at[i, "Close"])
        prev_max200 = df.at[i - 1, "Max200"]
        prev_min100 = df.at[i - 1, "Min100"]
        prev_atr = df.at[i - 1, "ATR25"]

        # Guard NaNs
        if pd.isna(prev_max200) or pd.isna(prev_min100) or pd.isna(prev_atr):
            # cannot make decisions without previous day's indicators
            if in_trade:
                # keep existing trade if indicators missing (still evaluate stop vs exit using stored entry_atr/price)
                pass
            else:
                df.at[i, "Signal"] = 0
                df.at[i, "Reason"] = "missing_prev"
                # EntryPrice / StopPrice / ExitPrice / PnL_Percent remain NaN
                continue

        if not in_trade:
            # Entry rule: today's close > yesterday's Max200
            if today_close > prev_max200:
                in_trade = True
                entry_price = today_close
                entry_atr = prev_atr  # ATR used is ATR of the day before entry
                stop_level = entry_price - 3.0 * float(entry_atr)

                df.at[i, "Signal"] = 1
                df.at[i, "Reason"] = "enter"
                df.at[i, "EntryPrice"] = entry_price
                df.at[i, "StopPrice"] = stop_level
                # ExitPrice / PnL_Percent remain NaN
            else:
                df.at[i, "Signal"] = 0
                df.at[i, "Reason"] = "no_long"
                # EntryPrice / StopPrice / ExitPrice / PnL_Percent remain NaN
        else:
            # If in trade, check stop-loss first (using ATR captured at entry)
            stop_level = entry_price - 3.0 * float(entry_atr)
            if today_close < stop_level:
                in_trade = False
                df.at[i, "Signal"] = 0
                df.at[i, "Reason"] = "stop_loss"
                # trade closed -> clear debug columns (maintain until closed)
                df.at[i, "EntryPrice"] = np.nan
                df.at[i, "StopPrice"] = np.nan
                df.at[i, "ExitPrice"] = today_close
                df.at[i, "PnL_Percent"] = ((today_close - entry_price) / entry_price) * 100
                entry_price = None
                entry_atr = None
            # Then check Min100 exit (uses yesterday's Min100)
            elif today_close < prev_min100:
                in_trade = False
                df.at[i, "Signal"] = 0
                df.at[i, "Reason"] = "exit_min100"
                # trade closed -> clear debug columns
                df.at[i, "EntryPrice"] = np.nan
                df.at[i, "StopPrice"] = np.nan
                df.at[i, "ExitPrice"] = today_close
                df.at[i, "PnL_Percent"] = ((today_close - entry_price) / entry_price) * 100
                entry_price = None
                entry_atr = None
            else:
                # still in trade -> maintain entry/stop values
                df.at[i, "Signal"] = 1
                df.at[i, "Reason"] = "hold"
                df.at[i, "EntryPrice"] = entry_price
                df.at[i, "StopPrice"] = stop_level
                # ExitPrice / PnL_Percent remain NaN

    # write back
    if inplace:
        df.to_csv(csv_path, index=False)
    else:
        out_path = csv_path.with_suffix(".signals.csv")
        df.to_csv(out_path, index=False)

    return df

def process_all(directory: Path = INDICATORS_DIR):
    if not directory.exists():
        raise FileNotFoundError(f"{directory} does not exist")
    for p in directory.glob("*.csv"):
        try:
            # no per-file print here anymore
            process_file(p, inplace=True)
        except Exception as e:
            print(f"Failed {p.name}: {e}")

if __name__ == "__main__":
    process_all()