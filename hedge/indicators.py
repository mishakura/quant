import pandas as pd
import os

# Paths
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
indicators_dir = os.path.join(base_dir, 'indicators')

try:
    # Load VIX data (assuming VIX.csv has 'Date' and 'Price')
    vix_file = os.path.join(data_dir, 'VIX.csv')
    print(f"Loading VIX from {vix_file}")
    vix_df = pd.read_csv(vix_file)
    print("VIX columns:", vix_df.columns.tolist())  # Debug
    vix_df['Date'] = pd.to_datetime(vix_df['Date'])
    vix_df.set_index('Date', inplace=True)
    vix_price = vix_df['Price']
    print("VIX loaded")

    # Combine into a single DataFrame (only VIX_Price needed)
    combined_df = pd.DataFrame({
        'VIX_Price': vix_price
    })
    print("DataFrame combined")

    # Drop rows with NaN
    combined_df.dropna(inplace=True)
    print(f"After dropna, shape: {combined_df.shape}")

    # Output to CSV in indicators folder
    output_file = os.path.join(indicators_dir, 'vix_curve_indicators.csv')
    combined_df.to_csv(output_file)
    print(f"Indicators computed and saved to {output_file}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()