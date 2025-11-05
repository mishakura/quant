import os
from pathlib import Path
import pandas as pd

## Calculate trades and create masters.

# Directories
indicators_dir = Path(os.path.join(os.path.dirname(__file__), "indicators"))
trades_dir = Path(os.path.join(os.path.dirname(__file__), "trades"))
os.makedirs(trades_dir, exist_ok=True)

def filter_trades():
    if not indicators_dir.exists():
        raise FileNotFoundError(f"{indicators_dir} does not exist")
    
    all_trades = []
    
    for csv_path in indicators_dir.glob("*.csv"):
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        
        # Filter rows where Reason is 'enter', 'exit_min100', or 'stop_loss'
        filtered_df = df[df['Reason'].isin(['enter', 'exit_min100', 'stop_loss'])]
        
        if not filtered_df.empty:
            # Add Ticker column
            ticker = csv_path.stem.replace('_indicators', '')
            filtered_df['Ticker'] = ticker
            
            # Save individual to trades directory
            output_path = trades_dir / csv_path.name
            filtered_df.to_csv(output_path, index=False)
            
            print(f"Filtered {len(filtered_df)} trade rows from {csv_path.name} to {output_path.name}")
            
            # Collect for master
            all_trades.append(filtered_df)
        else:
            print(f"No trade rows in {csv_path.name}")
    
    # Create master CSV
    if all_trades:
        master_df = pd.concat(all_trades, ignore_index=True)
        master_df = master_df.sort_values(['Date', 'Ticker'])
        master_path = trades_dir / "master.csv"
        master_df.to_csv(master_path, index=False)
        print(f"Master CSV created with {len(master_df)} total trade rows at {master_path}")
    else:
        print("No trades found across all files.")

if __name__ == "__main__":
    filter_trades()