#!/usr/bin/env python3
"""
Dynamic allocation using EMA crossover on SPX.

Strategy:
- ema_fast = EMA(span=126)
- ema_slow = EMA(span=315)
- If ema_fast > ema_slow => fully invested in SPX (position=1)
- Else => position=0 (cash)

Outputs:
- prints performance summary (total return, annualized return, vol, Sharpe, max drawdown)
- writes `strategy_output.csv` in the same folder with details
"""
from pathlib import Path
import pandas as pd
import numpy as np
import math
import sys

# Configuration
EMA_FAST = 126
EMA_SLOW = 315
OUT_CSV = "strategy_output.csv"
DEFAULT_FILE = "SPX.csv"

def find_spx_file(start_dir: Path):
    # Try same directory first
    p = start_dir / DEFAULT_FILE
    if p.exists():
        return p
    # Try sibling known path(s)
    alt = start_dir.parent / "fail_hedge" / DEFAULT_FILE
    if alt.exists():
        return alt
    # Try to search upward for any SPX.csv (limited)
    cur = start_dir
    for _ in range(5):
        candidate = cur / DEFAULT_FILE
        if candidate.exists():
            return candidate
        candidate2 = cur / "fail_hedge" / DEFAULT_FILE
        if candidate2.exists():
            return candidate2
        cur = cur.parent
    raise FileNotFoundError(f"Could not find {DEFAULT_FILE} in {start_dir} or common locations.")

def infer_periods_per_year(dates: pd.DatetimeIndex):
    if len(dates) < 2:
        return 252
    diffs = dates.to_series().diff().dt.days.dropna()
    median_days = diffs.median()
    if median_days <= 2:
        return 252
    if median_days <= 9:
        return 52
    if median_days <= 40:
        return 12
    # quarterly or longer
    if median_days <= 120:
        return 4
    # fallback
    return 252

def max_drawdown(cum_returns):
    roll_max = cum_returns.cummax()
    drawdown = cum_returns / roll_max - 1.0
    return drawdown.min()

def summarize(name, returns, periods_per_year):
    # returns are simple period returns (not cumulative)
    returns = returns.dropna()
    total_periods = len(returns)
    if total_periods == 0:
        return {}
    cumulative = (1 + returns).prod() - 1
    years = total_periods / periods_per_year
    if years > 0:
        cagr = (1 + cumulative) ** (1 / years) - 1
    else:
        cagr = float("nan")
    ann_vol = returns.std() * math.sqrt(periods_per_year)
    sharpe = (cagr / ann_vol) if ann_vol and not math.isnan(cagr) else float("nan")
    cum_series = (1 + returns).cumprod()
    mdd = max_drawdown(cum_series)
    return {
        "Total Return": cumulative,
        "CAGR": cagr,
        "Annual Vol": ann_vol,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": mdd
    }

def main():
    here = Path(__file__).resolve().parent
    try:
        spx_file = find_spx_file(here)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    print(f"Loading SPX data from: {spx_file}")
    df = pd.read_csv(spx_file, parse_dates=["Date"], dayfirst=False)
    # Ensure Date sorted ascending
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    if "Close" not in df.columns:
        raise KeyError("SPX.csv must contain a 'Close' column")

    price = df["Close"].astype(float)

    # Compute EMAs (span = number of periods)
    ema_fast = price.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = price.ewm(span=EMA_SLOW, adjust=False).mean()

    df["Close"] = price
    df["EMA_fast"] = ema_fast
    df["EMA_slow"] = ema_slow

    # Generate signal: 1 if fast > slow else 0
    df["signal_raw"] = (df["EMA_fast"] > df["EMA_slow"]).astype(int)
    # To avoid lookahead, apply position next period (shift)
    df["position"] = df["signal_raw"].shift(1).fillna(0).astype(int)

    # Period returns
    df["returns"] = df["Close"].pct_change()

    # Strategy returns: position * returns
    df["strategy_returns"] = df["position"] * df["returns"]

    # Buy-and-hold returns (always invested)
    df["bh_returns"] = df["returns"].fillna(0)

    # Cumulative returns (starting at 1)
    df["cum_strategy"] = (1 + df["strategy_returns"].fillna(0)).cumprod()
    df["cum_bh"] = (1 + df["bh_returns"].fillna(0)).cumprod()
    df["cum_price_norm"] = df["Close"] / df["Close"].iloc[0]

    # Determine periods per year from index
    ppy = infer_periods_per_year(df.index)

    # Summaries
    strat_summary = summarize("Strategy", df["strategy_returns"], ppy)
    bh_summary = summarize("BuyAndHold", df["bh_returns"], ppy)

    print("\nPerformance Summary")
    print(f"- Data rows: {len(df)}    periods/year ~ {ppy}")
    print("\nStrategy (EMA cross):")
    for k, v in strat_summary.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nBuy and Hold (SPX):")
    for k, v in bh_summary.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save detailed output
    outpath = here / OUT_CSV
    df_out = df[["Close", "EMA_fast", "EMA_slow", "signal_raw", "position", "returns", "strategy_returns", "bh_returns", "cum_strategy", "cum_bh"]]
    df_out.to_csv(outpath)
    print(f"\nDetailed results written to: {outpath}")

if __name__ == "__main__":
    main()