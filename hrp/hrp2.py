import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

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

def correl_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i+1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix = sort_ix.reset_index(drop=True)
    return sort_ix.tolist()

def get_cluster_var(cov, cluster_items):
    cov_ = cov.loc[cluster_items, cluster_items]
    w_ = 1. / np.diag(cov_)
    w_ /= w_.sum()
    w_ = w_.reshape(-1, 1)
    return float(np.dot(np.dot(w_.T, cov_), w_).item())

def get_rec_bipart_hrp(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while len(clusters) > 0:
        clusters = [cluster[i:j] for cluster in clusters for i, j in ((0, len(cluster)//2), (len(cluster)//2, len(cluster))) if len(cluster) > 1]
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            left = cluster[:len(cluster)//2]
            right = cluster[len(cluster)//2:]
            var_left = get_cluster_var(cov, left)
            var_right = get_cluster_var(cov, right)
            alpha = 1 - var_left / (var_left + var_right)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return w / w.sum()

def get_rec_bipart_herc(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while len(clusters) > 0:
        clusters = [cluster[i:j] for cluster in clusters for i, j in ((0, len(cluster)//2), (len(cluster)//2, len(cluster))) if len(cluster) > 1]
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            left = cluster[:len(cluster)//2]
            right = cluster[len(cluster)//2:]
            w[left] *= 0.5
            w[right] *= 0.5
    return w / w.sum()

# Calculate distance matrix and linkage
dist = correl_dist(corr)
link = linkage(dist, 'single')
sort_ix = get_quasi_diag(link)
sorted_tickers = corr.index[sort_ix].tolist()

# Calculate HRP weights
hrp_weights = get_rec_bipart_hrp(cov, sorted_tickers)
hrp_weights = hrp_weights.reindex(corr.index)  # Ensure original order

print(hrp_weights)

# Save to Excel
hrp_weights.to_frame('HRP_Weight').to_excel('hrp_weights.xlsx')

# Optional: Plot dendrogram
plt.figure(figsize=(8, 4))
dendrogram(link, labels=sorted_tickers, leaf_rotation=90)
plt.title("Asset Dendrogram (HRP)")
plt.tight_layout()
plt.show()

# Calculate HERC weights
herc_weights = get_rec_bipart_herc(cov, sorted_tickers)
herc_weights = herc_weights.reindex(corr.index)  # Ensure original order

print(herc_weights)

# Save to Excel
herc_weights.to_frame('HERC_Weight').to_excel('herc_weights.xlsx')
