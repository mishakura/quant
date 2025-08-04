import pandas as pd
import numpy as np
import riskfolio as rf

# Read correlation matrix
corr_df = pd.read_excel('corr.xlsx')
corr_df['Ticker'] = corr_df['Ticker'].astype(str).str.upper()
corr_df = corr_df.set_index('Ticker')
tickers = corr_df.index.tolist()
corr = corr_df.loc[tickers, tickers]

# Read volatility vector
vol_df = pd.read_excel('vol.xlsx')
vol_df['Asset'] = vol_df['Asset'].astype(str).str.upper()
vol_df['Vol'] = vol_df['Vol'].astype(str).str.replace('%', '').astype(float) / 100
vol_df = vol_df.set_index('Asset')
vol = vol_df['Vol'].reindex(corr.index).values

# Build covariance matrix
cov = np.outer(vol, vol) * corr.values
cov = pd.DataFrame(cov, index=corr.index, columns=corr.columns)

# Generate fake returns with the same covariance structure
np.random.seed(42)
returns = np.random.multivariate_normal(mean=np.zeros(len(cov)), cov=cov, size=100)
returns = pd.DataFrame(returns, columns=cov.columns)

ax = rf.plot_dendrogram(returns=returns,
                        codependence='pearson',
                        linkage='single',
                        k=None,
                        max_k=10,
                        leaf_order=True,
                        ax=None)

# HRP allocation using HCPortfolio (now HERC)
port = rf.HCPortfolio(returns=returns)
w = port.optimization(
    model='HERC',                # <-- Changed from 'HRP' to 'HERC'
    codependence='pearson',
    rm='MV',
    rf=0,
    linkage='single',
    max_k=10,
    leaf_order=True
)

print(w.T)

# Save HERC weights to Excel
w.T.to_excel("herc_weights.xlsx")
