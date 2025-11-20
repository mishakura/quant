import pandas as pd
import os

def compute_signals():
    indicators_folder = './indicators'
    signals_folder = './signals'
    os.makedirs(signals_folder, exist_ok=True)
    
    for file in os.listdir(indicators_folder):
        if file.endswith('.csv'):
            asset_name = file.replace('_indicators.csv', '')
            df = pd.read_csv(os.path.join(indicators_folder, file))
            
            # Initialize signal column
            df['signal'] = 0
            
            for i in range(len(df)):
                ema8 = df.loc[i, 'EMA8']
                ema32 = df.loc[i, 'EMA32']
                
                if ema8 < ema32:
                    df.loc[i, 'signal'] = 1
                else:
                    df.loc[i, 'signal'] = 0
            
            # Save to signals folder
            output_path = os.path.join(signals_folder, f'{asset_name}_signals.csv')
            df.to_csv(output_path, index=False)
            print(f'Signals computed for {asset_name}')

if __name__ == '__main__':
    compute_signals()