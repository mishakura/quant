import yfinance as yf
import pandas as pd
import numpy as np
from tickers import tickers

# Download monthly adjusted close prices for SPY (market) for the last 5 years
spy = yf.download('SPY', period='5y', interval='1mo', auto_adjust=True)
spy['Return'] = spy['Close'].pct_change()
market_ret = spy['Return']
results = []
for ticker in tickers:
    try:
        data = yf.download(ticker, period='5y', interval='1mo', auto_adjust=True)
        data['Return'] = data['Close'].pct_change()
        asset_ret = data['Return']
        # Align dates
        aligned = pd.DataFrame({'asset': asset_ret, 'market': market_ret}).dropna()
        if len(aligned) < 12:
            continue  # skip if not enough data
        cov = np.cov(aligned['asset'], aligned['market'])[0][1]
        var = np.var(aligned['market'])
        beta = cov / var if var != 0 else np.nan
        results.append({'Asset': ticker, 'Beta': beta})
    except Exception as e:
        results.append({'Asset': ticker, 'Beta': np.nan})

output_df = pd.DataFrame(results)
output_df = output_df.dropna(subset=['Beta'])
output_df['ReverseRank'] = output_df['Beta'].rank(ascending=True, method='min')
output_df = output_df.sort_values('ReverseRank')
output_df.to_excel('bab_betas.xlsx', index=False)
print('BAB betas and reverse rank saved to bab_betas.xlsx:')
print(output_df)
