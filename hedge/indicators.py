import os
import pandas as pd

# Function to calculate True Range
def true_range(high, low, prev_close):
    return max(high - low, abs(high - prev_close), abs(low - prev_close))

# Function to calculate ATR (Average True Range) over a period (default 25)
def calculate_atr(df, period=25):
    df['TR'] = df.apply(lambda row: true_range(row['high'], row['low'], df.loc[row.name - 1, 'close'] if row.name > 0 else 0), axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

# Function to calculate lowest low over last 200 days (previous)
def lowest_low_200(df):
    df['Lowest_Low_200'] = df['low'].shift(1).rolling(window=200).min()
    return df

# Function to calculate highest high over last 100 days (previous)
def highest_high_100(df):
    df['Highest_High_100'] = df['high'].shift(1).rolling(window=100).max()
    return df

# Function to calculate EMAs
def calculate_emas(df):
    df['EMA8'] = df['close'].ewm(span=8).mean()
    df['EMA32'] = df['close'].ewm(span=32).mean()
    return df

# Main function
def process_assets():
    data_folder = './data'
    indicators_folder = './indicators'
    os.makedirs(indicators_folder, exist_ok=True)
    
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            asset_name = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_folder, file))
            # Take first 4 columns and assign names
            df = df.iloc[:, :4]
            df.columns = ['date', 'close', 'high', 'low']
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate indicators
            df = lowest_low_200(df)
            df = calculate_atr(df)
            df = highest_high_100(df)
            df = calculate_emas(df)
            
            # Save to indicators folder
            output_path = os.path.join(indicators_folder, f'{asset_name}_indicators.csv')
            df.to_csv(output_path, index=False)
            print(f'Processed {asset_name}')

if __name__ == '__main__':
    process_assets()