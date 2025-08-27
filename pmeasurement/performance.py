import pandas as pd
import os

# File paths
RETURNS_FILE = 'bd_data_merged.xlsx'
WEIGHTS_FILE = 'weights.xlsx'
RETURNS_SHEET = 'USD_RETURNS'
PORTFOLIO_RETURNS_SHEET = 'PORTFOLIO_RETURNS'

def main():
    weights_xls = pd.ExcelFile(WEIGHTS_FILE)

    # --- Load full returns data (for YTD Full and Last 5 Days Return) ---
    returns_full = pd.read_excel(RETURNS_FILE, sheet_name=RETURNS_SHEET)
    returns_full.set_index('FECHA', inplace=True)
    returns_full.index = pd.to_datetime(returns_full.index)

    # --- Compute YTD return per asset with weight > 0 in any portfolio ---
    # Find all assets with weight > 0 in any portfolio
    assets_with_weight = set()
    for sheet in weights_xls.sheet_names:
        weights = pd.read_excel(WEIGHTS_FILE, sheet_name=sheet)
        for col in weights.columns:
            if isinstance(col, str) and col.strip().lower() == 'fecha':
                continue
            if (weights[col] > 0).any():
                assets_with_weight.add(col)


    # Read returns (for quarter-end trimmed calculations)
    returns = returns_full.copy()

    # --- Trim data to last available quarter-end ---
    # Find the last date that is a quarter-end (March 31, June 30, Sept 30, Dec 31)
    def is_quarter_end(date):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return (date.month, date.day) in [(3,31), (6,30), (9,30), (12,31)]

    # Ensure index is datetime
    returns.index = pd.to_datetime(returns.index)
    # Find all quarter-end dates in the index
    quarter_ends = [d for d in returns.index if is_quarter_end(d)]
    if not quarter_ends:
        raise ValueError("No quarter-end dates found in returns index.")
    last_quarter_end = max(quarter_ends)
    # Only use data up to and including last_quarter_end
    returns = returns.loc[returns.index <= last_quarter_end]

    # Calculate YTD return for each asset (using trimmed data)
    asset_ytd = {}
    for asset in assets_with_weight:
        if asset in returns.columns:
            asset_ret = returns[asset].dropna()
            n = asset_ret.shape[0]
            ytd = (1 + asset_ret).prod() - 1 if n > 0 else float('nan')
            asset_ytd[asset] = ytd
    asset_ytd_df = pd.DataFrame(list(asset_ytd.items()), columns=['Asset', 'YTD'])
    portfolio_returns = pd.DataFrame(index=returns.index)

    for sheet in weights_xls.sheet_names:
        weights = pd.read_excel(WEIGHTS_FILE, sheet_name=sheet)
        # Accept both 'FECHA' and 'fecha' for robustness
        date_col = None
        for col in weights.columns:
            if isinstance(col, str) and col.strip().lower() == 'fecha':
                date_col = col
                break
        if not date_col:
            continue
        weights[date_col] = pd.to_datetime(weights[date_col])
        # Only use weights up to last_quarter_end
        weights = weights[weights[date_col] <= last_quarter_end]
        weights.set_index(date_col, inplace=True)
        # Align columns and index
        common_assets = weights.columns.intersection(returns.columns)
        aligned_weights = weights[common_assets].reindex(index=returns.index).fillna(0)
        aligned_returns = returns[common_assets].reindex(index=returns.index).fillna(0)
        # Calculate daily portfolio return
        daily_ret = (aligned_weights * aligned_returns).sum(axis=1)
        portfolio_returns[sheet] = daily_ret


    # Calculate stats per portfolio
    stats = []
    if 'SPY' not in returns.columns:
        print("SPY column not found in returns. Beta/Alpha will be NaN.")
        spy_returns = pd.Series(index=returns.index, dtype=float)
    else:
        spy_returns = returns['SPY']

    # Calculate SPY YTD for comparison
    spy_ytd = (1 + spy_returns.dropna()).prod() - 1 if not spy_returns.dropna().empty else float('nan')
    spy_n = spy_returns.dropna().shape[0]

    risk_free_annual = 0.05
    risk_free_daily = risk_free_annual / 252

    for portfolio in list(portfolio_returns.columns) + ['SPY']:
        if portfolio == 'SPY':
            ret = spy_returns
            ret_full = returns_full['SPY'] if 'SPY' in returns_full.columns else pd.Series(dtype=float)
        else:
            ret = portfolio_returns[portfolio]
            # Recalculate the full portfolio return using all dates in returns_full
            # Align weights to returns_full
            weights = pd.read_excel(WEIGHTS_FILE, sheet_name=portfolio)
            date_col = None
            for col in weights.columns:
                if isinstance(col, str) and col.strip().lower() == 'fecha':
                    date_col = col
                    break
            if date_col:
                weights[date_col] = pd.to_datetime(weights[date_col])
                weights.set_index(date_col, inplace=True)
                # Use all available dates up to today
                common_assets = weights.columns.intersection(returns_full.columns)
                aligned_weights = weights[common_assets].reindex(index=returns_full.index).fillna(0)
                aligned_returns = returns_full[common_assets].reindex(index=returns_full.index).fillna(0)
                ret_full = (aligned_weights * aligned_returns).sum(axis=1)
            else:
                ret_full = pd.Series(dtype=float)

        ret_clean = ret.dropna()
        # Q2 2025 Return
        q2_start = '2025-04-01'
        q2_end = '2025-06-30'
        q2 = ret_clean.loc[(ret_clean.index >= q2_start) & (ret_clean.index <= q2_end)]
        q2_return = (1 + q2).prod() - 1 if not q2.empty else float('nan')
        # Standard deviation (annualized)
        n = ret_clean.shape[0]
        stdev = ret_clean.std() * (252 ** 0.5) if n > 1 else float('nan')
        # Mean return (daily, not annualized)
        mean = ret_clean.mean() if n > 0 else float('nan')
        # YTD return (from first to last non-NaN value)
        ytd = (1 + ret_clean).prod() - 1 if n > 0 else float('nan')
        # Annualized YTD
        ytd_ann = (1 + ytd) ** (252 / n) - 1 if n > 0 else float('nan')
        # --- YTD (Full): Use all available data up to today ---
        ret_full_clean = ret_full.dropna()
        n_full = ret_full_clean.shape[0]
        ytd_full = (1 + ret_full_clean).prod() - 1 if n_full > 0 else float('nan')
        # --- Last 5 Days Return ---
        if n_full >= 5:
            last5 = ret_full_clean.iloc[-5:]
            last5_return = (1 + last5).prod() - 1
        elif n_full > 0:
            last5 = ret_full_clean
            last5_return = (1 + last5).prod() - 1
        else:
            last5_return = float('nan')
        # Max Drawdown
        cumulative = (1 + ret_clean).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if n > 0 else float('nan')
        # Sharpe Ratio (risk-free = 5%)
        excess = ret_clean - risk_free_daily
        sharpe = (excess.mean() / ret_clean.std()) * (252 ** 0.5) if ret_clean.std() != 0 and n > 1 else float('nan')
        # Sortino Ratio (risk-free = 5%)
        downside = excess[excess < 0]
        sortino = (excess.mean() / downside.std()) * (252 ** 0.5) if downside.std() != 0 and n > 1 else float('nan')
        # Skewness
        skew = ret_clean.skew() if n > 2 else float('nan')
        # Kurtosis
        kurt = ret_clean.kurtosis() if n > 3 else float('nan')
        # Win Rate
        win_rate = (ret_clean > 0).sum() / n if n > 0 else float('nan')
        # Best/Worst Day
        best_day = ret_clean.max() if n > 0 else float('nan')
        worst_day = ret_clean.min() if n > 0 else float('nan')
        # Volatility (daily, not annualized)
        vol_daily = ret_clean.std() if n > 1 else float('nan')
        # Tracking Error and Information Ratio (vs SPY, skip for SPY itself)
        if portfolio != 'SPY' and spy_returns.dropna().shape[0] > 1:
            aligned = pd.concat([ret_clean, spy_returns], axis=1, join='inner').dropna()
            diff = aligned.iloc[:,0] - aligned.iloc[:,1]
            tracking_error = diff.std() * (252 ** 0.5) if diff.shape[0] > 1 else float('nan')
            # Information Ratio: mean of excess return over risk-free minus SPY excess, divided by tracking error
            spy_excess = aligned.iloc[:,1] - risk_free_daily
            info_ratio = ((excess.loc[aligned.index] - spy_excess).mean() / tracking_error) if tracking_error != 0 and diff.shape[0] > 1 else float('nan')
        else:
            tracking_error = float('nan')
            info_ratio = float('nan')
        # Beta and alpha (CAPM)
        if portfolio == 'SPY':
            beta = 1.0
            alpha = 0.0
        elif spy_returns.dropna().empty or ret_clean.empty:
            beta = float('nan')
            alpha = float('nan')
        else:
            aligned = pd.concat([ret_clean, spy_returns], axis=1, join='inner').dropna()
            if aligned.shape[0] < 2:
                beta = float('nan')
                alpha = float('nan')
            else:
                cov = aligned.iloc[:,0].cov(aligned.iloc[:,1])
                var = aligned.iloc[:,1].var()
                beta = cov / var if var != 0 else float('nan')
                alpha = aligned.iloc[:,0].mean() - beta * aligned.iloc[:,1].mean()
        # YTD - SPY (not annualized)
        ytd_minus_spy = ytd - spy_ytd if n > 0 and spy_n > 0 else float('nan')
        stats.append({
            'Portfolio': portfolio,
            'Mean (daily)': mean,
            'Stdev (annualized)': stdev,
            'Volatility (daily)': vol_daily,
            'YTD': ytd,
            'YTD (annualized)': ytd_ann,
            'YTD (Full)': ytd_full,                  # <--- NEW
            'Last 5 Days Return': last5_return,      # <--- NEW
            'YTD - SPY': ytd_minus_spy,
            'Q2 2025 Return': q2_return,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Skewness': skew,
            'Kurtosis': kurt,
            'Win Rate': win_rate,
            'Best Day': best_day,
            'Worst Day': worst_day,
            'Tracking Error': tracking_error,
            'Information Ratio': info_ratio,
            'Beta (vs SPY)': beta,
            'Alpha (vs SPY)': alpha
        })

    stats_df = pd.DataFrame(stats)

    # Write the new sheets to the existing Excel file
    with pd.ExcelWriter(RETURNS_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        portfolio_returns.to_excel(writer, sheet_name=PORTFOLIO_RETURNS_SHEET)
        stats_df.to_excel(writer, sheet_name='PORTFOLIO_STATS', index=False)
        asset_ytd_df.to_excel(writer, sheet_name='ASSET_YTD', index=False)


if __name__ == '__main__':
    main()
