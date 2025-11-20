import pandas as pd
import os

# Paths
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
indicators_dir = os.path.join(base_dir, 'indicators')

try:
    # Load VXX data (assuming vxx_data.csv has 'Date' and 'Price')
    vxx_file = os.path.join(data_dir, 'vxx_data.csv')
    print(f"Loading VXX from {vxx_file}")
    vxx_df = pd.read_csv(vxx_file)
    print("VXX columns:", vxx_df.columns.tolist())  # Debug
    vxx_df['Date'] = pd.to_datetime(vxx_df['Date'])
    vxx_df.set_index('Date', inplace=True)
    vxx_price = vxx_df['Price']
    print("VXX loaded")

    # Combine into a single DataFrame (only VXX_Price needed)
    combined_df = pd.DataFrame({
        'VXX_Price': vxx_price
    })
    print("DataFrame combined")

    # Calculate EMA8 and EMA32
    combined_df['EMA8'] = combined_df['VXX_Price'].ewm(span=8).mean()
    combined_df['EMA32'] = combined_df['VXX_Price'].ewm(span=32).mean()

    # Drop rows with NaN
    combined_df.dropna(inplace=True)
    print(f"After dropna, shape: {combined_df.shape}")

    # Output to CSV in indicators folder
    output_file = os.path.join(indicators_dir, 'vxx_indicators.csv')
    combined_df.to_csv(output_file)
    print(f"Indicators computed and saved to {output_file}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()