import pandas as pd
import numpy as np
import os

from quant.utils.constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from quant.utils.dates import last_quarter_end
from quant.analytics.performance import (
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    return_skewness,
    return_kurtosis,
    win_rate as compute_win_rate,
    beta as compute_beta,
    tracking_error as compute_tracking_error,
    information_ratio as compute_information_ratio,
)

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
    returns.index = pd.to_datetime(returns.index)
    last_qe = last_quarter_end(returns.index)
    # Only use data up to and including last_quarter_end
    returns = returns.loc[returns.index <= last_qe]

    # Calculate YTD return for each asset (using trimmed data)
    asset_ytd = {}
    for asset in assets_with_weight:
        if asset in returns.columns:
            asset_ret = returns[asset].dropna()
            n = asset_ret.shape[0]
            ytd = (1 + asset_ret).prod() - 1 if n > 0 else float('nan')
            asset_ytd[asset] = ytd
    asset_ytd_df = pd.DataFrame(list(asset_ytd.items()), columns=['Asset', 'YTD'])
    portfolio_returns = pd.DataFrame(index=returns_full.index)

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
        weights.set_index(date_col, inplace=True)
        # Align columns and index
        common_assets = weights.columns.intersection(returns_full.columns)
        aligned_weights = weights[common_assets].reindex(index=returns_full.index).fillna(0)
        aligned_returns = returns_full[common_assets].reindex(index=returns_full.index).fillna(0)
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

    risk_free_annual = DEFAULT_RISK_FREE_RATE
    risk_free_daily = risk_free_annual / TRADING_DAYS_PER_YEAR

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
        stdev = annualized_volatility(ret_clean) if n > 1 else float('nan')
        # Mean return (daily, not annualized)
        mean = ret_clean.mean() if n > 0 else float('nan')
        # YTD return (from first to last non-NaN value)
        ytd = (1 + ret_clean).prod() - 1 if n > 0 else float('nan')
        # Annualized YTD
        ytd_ann = (1 + ytd) ** (TRADING_DAYS_PER_YEAR / n) - 1 if n > 0 else float('nan')
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
        mdd = max_drawdown(ret_clean) if n > 0 else float('nan')
        # Sharpe Ratio
        sharpe = sharpe_ratio(ret_clean, risk_free_rate=risk_free_annual) if ret_clean.std() != 0 and n > 1 else float('nan')
        # Sortino Ratio
        sortino = sortino_ratio(ret_clean, risk_free_rate=risk_free_annual) if n > 1 else float('nan')
        # Skewness
        skew = return_skewness(ret_clean) if n > 2 else float('nan')
        # Kurtosis
        kurt = return_kurtosis(ret_clean) if n > 3 else float('nan')
        # Win Rate
        wr = compute_win_rate(ret_clean) if n > 0 else float('nan')
        # Best/Worst Day
        best_day = ret_clean.max() if n > 0 else float('nan')
        worst_day = ret_clean.min() if n > 0 else float('nan')
        # Volatility (daily, not annualized)
        vol_daily = ret_clean.std() if n > 1 else float('nan')
        # Tracking Error and Information Ratio (vs SPY, skip for SPY itself)
        if portfolio != 'SPY' and spy_returns.dropna().shape[0] > 1:
            te = compute_tracking_error(ret_clean, spy_returns)
            ir = compute_information_ratio(ret_clean, spy_returns)
        else:
            te = float('nan')
            ir = float('nan')
        # Beta and alpha (CAPM)
        if portfolio == 'SPY':
            b = 1.0
            alpha = 0.0
        elif spy_returns.dropna().empty or ret_clean.empty:
            b = float('nan')
            alpha = float('nan')
        else:
            b = compute_beta(ret_clean, spy_returns)
            # Alpha: regression intercept (daily, not annualized)
            aligned = pd.concat([ret_clean, spy_returns], axis=1, join='inner').dropna()
            if aligned.shape[0] < 2 or np.isnan(b):
                alpha = float('nan')
            else:
                alpha = aligned.iloc[:,0].mean() - b * aligned.iloc[:,1].mean()
        # YTD - SPY (not annualized)
        ytd_minus_spy = ytd - spy_ytd if n > 0 and spy_n > 0 else float('nan')
        stats.append({
            'Portfolio': portfolio,
            'Mean (daily)': mean,
            'Stdev (annualized)': stdev,
            'Volatility (daily)': vol_daily,
            'YTD': ytd,
            'YTD (annualized)': ytd_ann,
            'YTD (Full)': ytd_full,
            'Last 5 Days Return': last5_return,
            'YTD - SPY': ytd_minus_spy,
            'Q2 2025 Return': q2_return,
            'Max Drawdown': mdd,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Skewness': skew,
            'Kurtosis': kurt,
            'Win Rate': wr,
            'Best Day': best_day,
            'Worst Day': worst_day,
            'Tracking Error': te,
            'Information Ratio': ir,
            'Beta (vs SPY)': b,
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
