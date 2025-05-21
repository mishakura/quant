import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from scipy import stats

def get_financial_metrics(ticker_symbol):
    """Fetch financial metrics for a given ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get current date and calculate date 8 years ago
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*8)
        
        # Get TTM income statement for Operating Income
        income_ttm = ticker.get_income_stmt(freq="trailing")
        
        # Get annual financial data for the last 8 years
        income_annual = ticker.get_income_stmt(freq="yearly")
        balance_annual = ticker.get_balance_sheet(freq="yearly")
        cashflow_annual = ticker.get_cashflow(freq="yearly")
        
        # Get quarterly data for TTM calculations
        income_quarterly = ticker.get_income_stmt(freq="quarterly")
        balance_quarterly = ticker.get_balance_sheet(freq="quarterly")
        
        # Get info dictionary for Enterprise Value
        info = ticker.info
        
        # Get operating income (TTM)
        operating_income = None
        if 'OperatingIncome' in income_ttm.index:
            operating_income = income_ttm.loc["OperatingIncome"].iloc[0]
        
        # Get enterprise value directly
        enterprise_value = info.get('enterpriseValue', None)
        
        # Calculate Operating Income / EV ratio
        operating_income_to_ev = None
        if operating_income is not None and enterprise_value is not None and enterprise_value > 0:
            operating_income_to_ev = operating_income / enterprise_value
        
        # Get up to 8 years of data (or what's available)
        years = min(8, len(income_annual.columns))
        
        # 1. Eight-Year Return on Assets (Geometric Average)
        roa_values = []
        for i in range(years):
            if i < len(income_annual.columns) and i < len(balance_annual.columns):
                year_col = income_annual.columns[i]
                # Ensure we're using the same year column from both dataframes
                if year_col in balance_annual.columns:
                    net_income = income_annual.loc["NetIncome"][year_col] if "NetIncome" in income_annual.index else None
                    total_assets = balance_annual.loc["TotalAssets"][year_col] if "TotalAssets" in balance_annual.index else None
                    
                    if pd.notna(net_income) and pd.notna(total_assets) and total_assets != 0:
                        roa_values.append(net_income / total_assets)
        
        eight_year_roa = None
        if len(roa_values) >= 3:  # Require at least 3 years of data for a meaningful average
            # Calculate geometric mean using all values (including negative ones)
            # For financial ratios, we use (1+r) approach to handle negative values
            roa_for_calc = [(1 + r) for r in roa_values]
            eight_year_roa = np.prod(np.array(roa_for_calc)) ** (1 / len(roa_values)) - 1
        
        # 2. Eight-Year Return on Capital (Geometric Average)
        roc_values = []
        for i in range(years):
            if i < len(income_annual.columns) and i < len(balance_annual.columns):
                year_col = income_annual.columns[i]
                # Ensure we're using the same year column from both dataframes
                if year_col in balance_annual.columns:
                    operating_income_annual = income_annual.loc["OperatingIncome"][year_col] if "OperatingIncome" in income_annual.index else None
                    
                    # Calculate capital (Total Assets - Current Liabilities)
                    total_assets = balance_annual.loc["TotalAssets"][year_col] if "TotalAssets" in balance_annual.index else None
                    current_liabilities = balance_annual.loc["CurrentLiabilities"][year_col] if "CurrentLiabilities" in balance_annual.index else None
                    
                    if pd.notna(operating_income_annual) and pd.notna(total_assets) and pd.notna(current_liabilities):
                        capital = total_assets - current_liabilities
                        if capital > 0:  # Only calculate if capital is positive
                            roc_values.append(operating_income_annual / capital)
        
        eight_year_roc = None
        if len(roc_values) >= 3:  # Require at least 3 years of data for a meaningful average
            # Calculate geometric mean using all values
            roc_for_calc = [(1 + r) for r in roc_values]
            eight_year_roc = np.prod(np.array(roc_for_calc)) ** (1 / len(roc_values)) - 1
        
        # 3. Sum (eight-year FCF) / total assets (t)
        fcf_sum = 0
        current_total_assets = None
        
        # Use specific year columns to ensure data consistency
        if "FreeCashFlow" in cashflow_annual.index and len(cashflow_annual.columns) > 0:
            fcf_values = [cashflow_annual.loc["FreeCashFlow"][year] for year in cashflow_annual.columns[:years]
                         if pd.notna(cashflow_annual.loc["FreeCashFlow"][year])]
            fcf_sum = sum(fcf_values)
        
        if "TotalAssets" in balance_annual.index and len(balance_annual.columns) > 0:
            current_total_assets = balance_annual.loc["TotalAssets"].iloc[0]  # Most recent total assets
        
        fcf_sum_to_assets = None
        if fcf_sum != 0 and pd.notna(current_total_assets) and current_total_assets != 0:
            fcf_sum_to_assets = fcf_sum / current_total_assets
        
        # 4. Eight-year gross margin growth (geometric average) and Margin Stability
        gross_margins = []
        
        for i in range(years):
            if i < len(income_annual.columns):
                year_col = income_annual.columns[i]
                gross_profit = income_annual.loc["GrossProfit"][year_col] if "GrossProfit" in income_annual.index else None
                total_revenue = income_annual.loc["TotalRevenue"][year_col] if "TotalRevenue" in income_annual.index else None
                
                if pd.notna(gross_profit) and pd.notna(total_revenue) and total_revenue != 0:
                    gross_margins.append(gross_profit / total_revenue)
        
        # Calculate Margin Stability (MS) = Average Gross Margin / Standard Deviation of Gross Margin
        margin_stability = None
        if len(gross_margins) >= 3:  # Need at least 3 years to calculate a meaningful std dev
            avg_gross_margin = np.mean(gross_margins)
            std_gross_margin = np.std(gross_margins, ddof=1)  # Use sample standard deviation
            
            if std_gross_margin > 0:  # Avoid division by zero
                margin_stability = avg_gross_margin / std_gross_margin
        
        gross_margin_growth = None
        if len(gross_margins) > 1:
            # Ensure we calculate growth rates correctly (older year / newer year)
            growth_rates = []
            for i in range(len(gross_margins)-1):
                # Gross margin in year i compared to gross margin in year i+1 (which is an earlier year)
                # This is because data is ordered from most recent to oldest
                if gross_margins[i+1] != 0:  # Avoid division by zero
                    growth_rate = (gross_margins[i] / gross_margins[i+1]) - 1
                    if pd.notna(growth_rate) and np.isfinite(growth_rate):
                        growth_rates.append(growth_rate)
            
            if len(growth_rates) >= 2:  # Need at least 2 growth rates for meaningful average
                # Calculate geometric mean of growth rates
                growth_rates_for_calc = [(1 + r) for r in growth_rates]
                gross_margin_growth = np.prod(np.array(growth_rates_for_calc)) ** (1 / len(growth_rates)) - 1
        
        # NEW METRIC 1: Current ROA = Net income before extraordinary items (t) / total assets (t)
        current_roa = None
        fs_roa = 0  # Default to 0
        
        if len(income_annual.columns) > 0 and len(balance_annual.columns) > 0:
            most_recent_year = income_annual.columns[0]  # Most recent year
            
            # Get net income for most recent year
            net_income_current = income_annual.loc["NetIncome"][most_recent_year] if "NetIncome" in income_annual.index else None
            
            # Get total assets for most recent year
            if most_recent_year in balance_annual.columns:
                total_assets_current = balance_annual.loc["TotalAssets"][most_recent_year] if "TotalAssets" in balance_annual.index else None
                
                # Calculate current ROA
                if pd.notna(net_income_current) and pd.notna(total_assets_current) and total_assets_current != 0:
                    current_roa = net_income_current / total_assets_current
                    
                    # Calculate FS_ROA
                    fs_roa = 1 if current_roa > 0 else 0
        
        # NEW METRIC 2: Current FCFTA = free cash flow (t) / total assets (t)
        current_fcfta = None
        fs_fcfta = 0  # Default to 0
        
        if len(cashflow_annual.columns) > 0 and len(balance_annual.columns) > 0:
            most_recent_year = cashflow_annual.columns[0]  # Most recent year
            
            # Get free cash flow for most recent year
            fcf_current = cashflow_annual.loc["FreeCashFlow"][most_recent_year] if "FreeCashFlow" in cashflow_annual.index else None
            
            # Get total assets for most recent year (ensure same year if possible)
            if most_recent_year in balance_annual.columns:
                total_assets_current = balance_annual.loc["TotalAssets"][most_recent_year] if "TotalAssets" in balance_annual.index else None
            else:
                # If years don't align perfectly, use the most recent total assets
                total_assets_current = balance_annual.loc["TotalAssets"].iloc[0] if "TotalAssets" in balance_annual.index else None
            
            # Calculate current FCFTA
            if pd.notna(fcf_current) and pd.notna(total_assets_current) and total_assets_current != 0:
                current_fcfta = fcf_current / total_assets_current
                
                # Calculate FS_FCFTA
                fs_fcfta = 1 if current_fcfta > 0 else 0
        
        # NEW METRIC 3: ACCRUAL = FCFTA - ROA
        accrual = None
        fs_accrual = 0  # Default to 0
        
        if pd.notna(current_fcfta) and pd.notna(current_roa):
            accrual = current_fcfta - current_roa
            
            # Calculate FS_ACCRUAL
            fs_accrual = 1 if accrual > 0 else 0
        
        # NEW METRIC 4: LEVER = long-term debt (t − 1) / total assets (t − 1) − long-term debt (t) / total assets (t)
        # Using quarterly data, where t is latest quarter and t-1 is same quarter from previous year
        lever = None
        fs_lever = 0  # Default to 0
        
        if len(balance_quarterly.columns) >= 4:  # Need at least 4 quarters of data to compare year-over-year
            current_quarter = balance_quarterly.columns[0]  # Most recent quarter
            year_ago_quarter = balance_quarterly.columns[4] if len(balance_quarterly.columns) >= 5 else balance_quarterly.columns[-1]  # Same quarter 1 year ago
            
            # Get long-term debt and total assets for current and previous year quarters
            lt_debt_current = balance_quarterly.loc["LongTermDebt"][current_quarter] if "LongTermDebt" in balance_quarterly.index else None
            total_assets_current = balance_quarterly.loc["TotalAssets"][current_quarter] if "TotalAssets" in balance_quarterly.index else None
            
            lt_debt_previous = balance_quarterly.loc["LongTermDebt"][year_ago_quarter] if "LongTermDebt" in balance_quarterly.index else None
            total_assets_previous = balance_quarterly.loc["TotalAssets"][year_ago_quarter] if "TotalAssets" in balance_quarterly.index else None
            
            # Calculate LEVER
            if (pd.notna(lt_debt_current) and pd.notna(total_assets_current) and 
                pd.notna(lt_debt_previous) and pd.notna(total_assets_previous) and
                total_assets_current > 0 and total_assets_previous > 0):
                
                lt_debt_to_assets_current = lt_debt_current / total_assets_current
                lt_debt_to_assets_previous = lt_debt_previous / total_assets_previous
                
                lever = lt_debt_to_assets_previous - lt_debt_to_assets_current
                
                # Calculate FS_LEVER
                fs_lever = 1 if lever > 0 else 0
        
        # NEW METRIC 5: LIQUID = current ratio (t) − current ratio (t − 1)
        # Using quarterly data, where t is latest quarter and t-1 is same quarter from previous year
        liquid = None
        fs_liquid = 0  # Default to 0
        
        if len(balance_quarterly.columns) >= 4:  # Need at least 4 quarters of data to compare year-over-year
            current_quarter = balance_quarterly.columns[0]  # Most recent quarter
            year_ago_quarter = balance_quarterly.columns[4] if len(balance_quarterly.columns) >= 5 else balance_quarterly.columns[-1]  # Same quarter 1 year ago
            
            # Get current assets and current liabilities for current quarter and year-ago quarter
            current_assets_current = balance_quarterly.loc["CurrentAssets"][current_quarter] if "CurrentAssets" in balance_quarterly.index else None
            current_liab_current = balance_quarterly.loc["CurrentLiabilities"][current_quarter] if "CurrentLiabilities" in balance_quarterly.index else None
            
            current_assets_previous = balance_quarterly.loc["CurrentAssets"][year_ago_quarter] if "CurrentAssets" in balance_quarterly.index else None
            current_liab_previous = balance_quarterly.loc["CurrentLiabilities"][year_ago_quarter] if "CurrentLiabilities" in balance_quarterly.index else None
            
            # Calculate current ratios and LIQUID
            if (pd.notna(current_assets_current) and pd.notna(current_liab_current) and 
                pd.notna(current_assets_previous) and pd.notna(current_liab_previous) and
                current_liab_current > 0 and current_liab_previous > 0):
                
                current_ratio_current = current_assets_current / current_liab_current
                current_ratio_previous = current_assets_previous / current_liab_previous
                
                liquid = current_ratio_current - current_ratio_previous
                
                # Calculate FS_LIQUID
                fs_liquid = 1 if liquid > 0 else 0
        
        # NEW METRIC 6: ROA_Change = year-over-year change in ROA using annual data
        roa_change = None
        fs_roa_change = 0  # Default to 0
        
        # Calculate ROA using annual data (more likely to be available)
        if len(income_annual.columns) >= 2 and len(balance_annual.columns) >= 2:
            # Get data for the most recent year (year 0)
            current_year = income_annual.columns[0]
            previous_year = income_annual.columns[1]
            
            # Calculate ROA for current year
            current_net_income = income_annual.loc["NetIncome"][current_year] if "NetIncome" in income_annual.index else None
            current_total_assets = balance_annual.loc["TotalAssets"][current_year] if "TotalAssets" in balance_annual.index and current_year in balance_annual.columns else None
            
            # Calculate ROA for previous year
            previous_net_income = income_annual.loc["NetIncome"][previous_year] if "NetIncome" in income_annual.index else None
            previous_total_assets = balance_annual.loc["TotalAssets"][previous_year] if "TotalAssets" in balance_annual.index and previous_year in balance_annual.columns else None
            
            # Calculate current year ROA and previous year ROA
            current_roa_annual = None
            previous_roa_annual = None
            
            if pd.notna(current_net_income) and pd.notna(current_total_assets) and current_total_assets > 0:
                current_roa_annual = current_net_income / current_total_assets
            
            if pd.notna(previous_net_income) and pd.notna(previous_total_assets) and previous_total_assets > 0:
                previous_roa_annual = previous_net_income / previous_total_assets
            
            # Calculate year-over-year change in ROA
            if pd.notna(current_roa_annual) and pd.notna(previous_roa_annual):
                roa_change = current_roa_annual - previous_roa_annual
                
                # Calculate FS_ROA_Change
                fs_roa_change = 1 if roa_change > 0 else 0
        
        # NEW METRIC 7: FCFTA_Change = year-over-year change in FCFTA using annual data
        fcfta_change = None
        fs_fcfta_change = 0  # Default to 0
        
        # Calculate FCFTA using annual data
        if len(cashflow_annual.columns) >= 2 and len(balance_annual.columns) >= 2:
            # Get data for the most recent year (year 0)
            current_year = cashflow_annual.columns[0]
            previous_year = cashflow_annual.columns[1]
            
            # Get free cash flow for current year
            current_fcf = cashflow_annual.loc["FreeCashFlow"][current_year] if "FreeCashFlow" in cashflow_annual.index else None
            
            # Get total assets for current year (ensure same year if possible)
            current_assets = None
            if current_year in balance_annual.columns:
                current_assets = balance_annual.loc["TotalAssets"][current_year] if "TotalAssets" in balance_annual.index else None
            
            # Get free cash flow for previous year
            previous_fcf = cashflow_annual.loc["FreeCashFlow"][previous_year] if "FreeCashFlow" in cashflow_annual.index else None
            
            # Get total assets for previous year (ensure same year if possible)
            previous_assets = None
            if previous_year in balance_annual.columns:
                previous_assets = balance_annual.loc["TotalAssets"][previous_year] if "TotalAssets" in balance_annual.index else None
            
            # Calculate current year FCFTA and previous year FCFTA
            current_fcfta_annual = None
            previous_fcfta_annual = None
            
            if pd.notna(current_fcf) and pd.notna(current_assets) and current_assets > 0:
                current_fcfta_annual = current_fcf / current_assets
            
            if pd.notna(previous_fcf) and pd.notna(previous_assets) and previous_assets > 0:
                previous_fcfta_annual = previous_fcf / previous_assets
            
            # Calculate year-over-year change in FCFTA
            if pd.notna(current_fcfta_annual) and pd.notna(previous_fcfta_annual):
                fcfta_change = current_fcfta_annual - previous_fcfta_annual
                
                # Calculate FS_FCFTA_Change
                fs_fcfta_change = 1 if fcfta_change > 0 else 0
        
        return {
            'Ticker': ticker_symbol,
            'OperatingIncome_TTM': operating_income,
            'EnterpriseValue': enterprise_value,
            'OperatingIncome_EV_Ratio': operating_income_to_ev,
            'Eight_Year_ROA': eight_year_roa,
            'Eight_Year_ROC': eight_year_roc,
            'FCF_Sum_to_Assets': fcf_sum_to_assets,
            'Eight_Year_Gross_Margin_Growth': gross_margin_growth,
            'Margin_Stability': margin_stability,
            'Years_Data_Available': years,
            'Current_ROA': current_roa,
            'FS_ROA': fs_roa,
            'Current_FCFTA': current_fcfta,
            'FS_FCFTA': fs_fcfta,
            'ACCRUAL': accrual,
            'FS_ACCRUAL': fs_accrual,
            'LEVER': lever,
            'FS_LEVER': fs_lever,
            'LIQUID': liquid,
            'FS_LIQUID': fs_liquid,
            'ROA_Change': roa_change,
            'FS_ROA_Change': fs_roa_change,
            'FCFTA_Change': fcfta_change,
            'FS_FCFTA_Change': fs_fcfta_change,
        }
    
    except Exception as e:
        print(f"Error getting data for {ticker_symbol}: {e}")
        return {
            'Ticker': ticker_symbol,
            'OperatingIncome_TTM': None,
            'EnterpriseValue': None,
            'OperatingIncome_EV_Ratio': None,
            'Eight_Year_ROA': None,
            'Eight_Year_ROC': None,
            'FCF_Sum_to_Assets': None,
            'Eight_Year_Gross_Margin_Growth': None,
            'Margin_Stability': None,
            'Years_Data_Available': 0,
            'Current_ROA': None,
            'FS_ROA': 0,
            'Current_FCFTA': None,
            'FS_FCFTA': 0,
            'ACCRUAL': None,
            'FS_ACCRUAL': 0,
            'LEVER': None,
            'FS_LEVER': 0,
            'LIQUID': None,
            'FS_LIQUID': 0,
            'ROA_Change': None,
            'FS_ROA_Change': 0,
            'FCFTA_Change': None,
            'FS_FCFTA_Change': 0,
        }

# Main function to process all tickers
def process_all_tickers(tickers):
    results = []
    
    # Get financial metrics for each ticker with a delay to avoid rate limiting
    for ticker_symbol in tickers:
        print(f"Processing {ticker_symbol}...")
        metrics = get_financial_metrics(ticker_symbol)
        results.append(metrics)
        
        # Add a small delay to avoid hitting API limits
        time.sleep(1)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Define metrics that need percentile calculation
    metrics_for_percentile = [
        'OperatingIncome_EV_Ratio',  # Higher is better
        'Eight_Year_ROA',            # Higher is better
        'Eight_Year_ROC',            # Higher is better
        'FCF_Sum_to_Assets',         # Higher is better
        'Eight_Year_Gross_Margin_Growth',  # Higher is better
        'Margin_Stability',          # Higher is better
        'Current_ROA',               # Higher is better
        'Current_FCFTA',             # Higher is better
        'ACCRUAL',                   # Higher is better
        'LEVER',                     # Higher is better
        'LIQUID',                    # Higher is better
        'ROA_Change',                # Higher is better
        'FCFTA_Change'               # Higher is better - newly added
    ]
    
    # Calculate percentiles for each metric
    for metric in metrics_for_percentile:
        # Drop rows with NaN values for the current metric
        df_filtered = results_df.dropna(subset=[metric])
        
        if not df_filtered.empty:
            # Calculate percentile for each ticker
            metric_values = df_filtered[metric].values
            
            # For metrics where higher is better
            metric_percentiles = [stats.percentileofscore(metric_values, x) / 100 for x in metric_values]
            
            # Create a mapping from ticker to percentile
            ticker_to_percentile = dict(zip(df_filtered['Ticker'].values, metric_percentiles))
            
            # Add percentiles to the original dataframe
            results_df[f'{metric}_Percentile'] = results_df['Ticker'].map(ticker_to_percentile)
        else:
            # If no valid values, create an empty column
            results_df[f'{metric}_Percentile'] = None
    
    # Calculate Max Percentile between Margin Stability and Margin Growth
    # First ensure both percentile columns exist
    margin_stability_col = 'Margin_Stability_Percentile'
    margin_growth_col = 'Eight_Year_Gross_Margin_Growth_Percentile'
    
    if margin_stability_col in results_df.columns and margin_growth_col in results_df.columns:
        # Calculate the maximum percentile between the two
        results_df['Max_Margin_Metric_Percentile'] = results_df[[margin_stability_col, margin_growth_col]].max(axis=1)
        
        # Add information about which metric contributed to the max value
        def determine_max_source(row):
            if pd.isna(row[margin_stability_col]) and pd.isna(row[margin_growth_col]):
                return 'Neither'
            elif pd.isna(row[margin_stability_col]):
                return 'Growth'
            elif pd.isna(row[margin_growth_col]):
                return 'Stability'
            else:
                return 'Growth' if row[margin_growth_col] >= row[margin_stability_col] else 'Stability'
        
        results_df['Max_Margin_Source'] = results_df.apply(determine_max_source, axis=1)
        
        print("Added Max Margin Metric calculation")
    else:
        print("Warning: One or both margin percentile columns not found")
    
    # Calculate Franchise Power Percentile
    # This is the average of ROA percentile, ROC Percentile, Sum cash flow percentile, Max_Margin_Metric_Percentile
    franchise_power_columns = [
        'Eight_Year_ROA_Percentile',
        'Eight_Year_ROC_Percentile',
        'FCF_Sum_to_Assets_Percentile',
        'Max_Margin_Metric_Percentile'
    ]
    
    # Create a function to calculate average of available percentiles
    def calculate_franchise_power(row):
        values = [row[col] for col in franchise_power_columns if pd.notna(row[col])]
        return np.mean(values) if values else None
    
    # Apply the function to calculate Franchise Power Percentile
    results_df['Franchise_Power_Percentile'] = results_df.apply(calculate_franchise_power, axis=1)
    
    # Calculate Financial Strength Score (sum of all FS binary indicators)
    fs_columns = [
        'FS_ROA',
        'FS_FCFTA',
        'FS_ACCRUAL',
        'FS_LEVER',
        'FS_LIQUID',
        'FS_ROA_Change',
        'FS_FCFTA_Change'  # Added the new FS metric
    ]
    
    # Sum up the FS indicators to get a Financial Strength Score (range: 0-7)
    results_df['Financial_Strength_Score'] = results_df[fs_columns].sum(axis=1)
    
    # Display the results
    print(results_df.head())
    
    # Save to CSV
    results_df.to_csv('financial_metrics_extended.csv', index=False)
    
    return results_df

# Add a function to calculate composite scores
def calculate_composite_score(results_df, weights=None):
    """
    Calculate a composite score based on percentiles of various metrics
    
    Parameters:
    - results_df: DataFrame with financial metrics and their percentiles
    - weights: Dictionary mapping metric percentile names to their weights in the composite score
              If None, equal weights will be used for all metrics
    
    Returns:
    - Updated DataFrame with a composite score column
    """
    # Get all percentile columns
    percentile_cols = [col for col in results_df.columns if col.endswith('_Percentile')]
    
    if not percentile_cols:
        print("No percentile columns found!")
        return results_df
    
    # If no weights provided, use equal weights
    if weights is None:
        weights = {col: 1.0/len(percentile_cols) for col in percentile_cols}
    
    # Validate weights
    for col in percentile_cols:
        if col not in weights:
            print(f"Warning: No weight provided for {col}, using 0")
            weights[col] = 0
    
    # Calculate weighted average of percentiles
    results_df['Composite_Score'] = 0
    
    for col in percentile_cols:
        # Only use rows where this metric is not NaN
        mask = results_df[col].notna()
        results_df.loc[mask, 'Composite_Score'] += results_df.loc[mask, col] * weights[col]
    
    # Count how many metrics were used for each row
    results_df['Metrics_Used'] = results_df[percentile_cols].count(axis=1)
    
    # Normalize the composite score based on how many metrics were used
    # This ensures that stocks with missing metrics aren't unduly penalized
    denominator = 0
    for col in percentile_cols:
        denominator += weights[col]
    
    # Calculate the expected sum of weights if all metrics were present
    total_weight = sum(weights.values())
    
    # Adjust composite score by the ratio of actual weights used to total possible weights
    # Only adjust if metrics are missing and the denominator is not zero
    if denominator > 0:
        results_df['Composite_Score'] = results_df['Composite_Score'] * (total_weight / denominator)
    
    # Sort by composite score (descending)
    results_df_sorted = results_df.sort_values('Composite_Score', ascending=False)
    
    return results_df_sorted

if __name__ == "__main__":
    
    # For testing, you can use a smaller subset
    test_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX", "NVDA"]
    process_all_tickers(test_tickers)  # Use test list for testing