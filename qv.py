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
        
        # Get quarterly data for more current metrics
        income_quarterly = ticker.get_income_stmt(freq="quarterly")
        balance_quarterly = ticker.get_balance_sheet(freq="quarterly")
        cashflow_quarterly = ticker.get_cashflow(freq="quarterly")
        
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
        
        # REVISED APPROACH: Proper TTM calculation using quarterly data
        # NEW METRIC 1: Current ROA = Net income TTM / total assets (t)
        current_roa = None
        fs_roa = 0  # Default to 0
        
        # Try to use quarterly data first for TTM calculation
        if len(income_quarterly.columns) >= 4 and len(balance_quarterly.columns) > 0:
            # Get the 4 most recent quarters
            recent_quarters = income_quarterly.columns[:4]
            
            # Sum net income from the last 4 quarters
            ttm_net_income = 0
            for quarter in recent_quarters:
                if "NetIncome" in income_quarterly.index:
                    quarter_net_income = income_quarterly.loc["NetIncome"][quarter]
                    if pd.notna(quarter_net_income):
                        ttm_net_income += quarter_net_income
            
            # Get most recent total assets
            most_recent_quarter = balance_quarterly.columns[0]  # Most recent quarter with balance sheet
            total_assets_current = balance_quarterly.loc["TotalAssets"][most_recent_quarter] if "TotalAssets" in balance_quarterly.index else None
            
            # Calculate current ROA using TTM Net Income
            if pd.notna(ttm_net_income) and pd.notna(total_assets_current) and total_assets_current != 0:
                current_roa = ttm_net_income / total_assets_current
                fs_roa = 1 if current_roa > 0 else 0
        
        # Fall back to annual data if insufficient quarterly data is available
        if current_roa is None and len(income_annual.columns) > 0 and len(balance_annual.columns) > 0:
            most_recent_year = income_annual.columns[0]  # Most recent year
            
            # Get net income for most recent year
            net_income_current = income_annual.loc["NetIncome"][most_recent_year] if "NetIncome" in income_annual.index else None
            
            # Get total assets for most recent year
            if most_recent_year in balance_annual.columns:
                total_assets_current = balance_annual.loc["TotalAssets"][most_recent_year] if "TotalAssets" in balance_annual.index else None
                
                # Calculate current ROA
                if pd.notna(net_income_current) and pd.notna(total_assets_current) and total_assets_current != 0:
                    current_roa = net_income_current / total_assets_current
                    fs_roa = 1 if current_roa > 0 else 0
        
        # NEW METRIC 2: Current FCFTA = free cash flow TTM / total assets (t)
        current_fcfta = None
        fs_fcfta = 0  # Default to 0
        
        # Try to use quarterly data first for TTM calculation
        if len(cashflow_quarterly.columns) >= 4 and len(balance_quarterly.columns) > 0:
            # Get the 4 most recent quarters
            recent_quarters = cashflow_quarterly.columns[:4]
            
            # Sum free cash flow from the last 4 quarters
            ttm_fcf = 0
            for quarter in recent_quarters:
                if "FreeCashFlow" in cashflow_quarterly.index:
                    quarter_fcf = cashflow_quarterly.loc["FreeCashFlow"][quarter]
                    if pd.notna(quarter_fcf):
                        ttm_fcf += quarter_fcf
            
            # Get most recent total assets
            most_recent_quarter = balance_quarterly.columns[0]  # Most recent quarter with balance sheet
            total_assets_current = balance_quarterly.loc["TotalAssets"][most_recent_quarter] if "TotalAssets" in balance_quarterly.index else None
            
            # Calculate current FCFTA using TTM Free Cash Flow
            if pd.notna(ttm_fcf) and pd.notna(total_assets_current) and total_assets_current != 0:
                current_fcfta = ttm_fcf / total_assets_current
                fs_fcfta = 1 if current_fcfta > 0 else 0
        
        # Fall back to annual data if insufficient quarterly data is available
        if current_fcfta is None and len(cashflow_annual.columns) > 0 and len(balance_annual.columns) > 0:
            most_recent_year = cashflow_annual.columns[0]  # Most recent year
            
            # Get free cash flow for most recent year
            fcf_current = cashflow_annual.loc["FreeCashFlow"][most_recent_year] if "FreeCashFlow" in cashflow_annual.index else None
            
            # Get total assets for most recent year
            if most_recent_year in balance_annual.columns:
                total_assets_current = balance_annual.loc["TotalAssets"][most_recent_year] if "TotalAssets" in balance_annual.index else None
            else:
                # If years don't align perfectly, use the most recent total assets
                total_assets_current = balance_annual.loc["TotalAssets"].iloc[0] if "TotalAssets" in balance_annual.index else None
            
            # Calculate current FCFTA
            if pd.notna(fcf_current) and pd.notna(total_assets_current) and total_assets_current != 0:
                current_fcfta = fcf_current / total_assets_current
                fs_fcfta = 1 if current_fcfta > 0 else 0
        
        # NEW METRIC 3: ACCRUAL = FCFTA - ROA
        accrual = None
        fs_accrual = 0  # Default to 0
        
        if pd.notna(current_fcfta) and pd.notna(current_roa):
            accrual = current_fcfta - current_roa
            fs_accrual = 1 if accrual > 0 else 0
                
        # Add debug info for gross margins and margin stability
        print(f"- Gross Margins: {gross_margins}")
        print(f"- Margin Stability: {margin_stability}")
        
        # Add debug info for this ticker
        print(f"Debug for {ticker_symbol}:")
        print(f"- ROA values: {roa_values}")
        print(f"- ROC values: {roc_values}")
        print(f"- Eight-Year ROA: {eight_year_roa}")
        print(f"- Eight-Year ROC: {eight_year_roc}")
        print(f"- Current ROA: {current_roa}, FS_ROA: {fs_roa}")
        print(f"- Current FCFTA: {current_fcfta}, FS_FCFTA: {fs_fcfta}")
        print(f"- ACCRUAL: {accrual}, FS_ACCRUAL: {fs_accrual}")
        print(f"- Data source: {'Quarterly TTM' if len(income_quarterly.columns) >= 4 else 'Annual'}")
        print("-------------------")
        
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
            # Revised metrics
            'Current_ROA': current_roa,
            'FS_ROA': fs_roa,
            'Current_FCFTA': current_fcfta,
            'FS_FCFTA': fs_fcfta,
            'ACCRUAL': accrual,
            'FS_ACCRUAL': fs_accrual,
            'Data_Source': 'Quarterly TTM' if len(income_quarterly.columns) >= 4 else 'Annual',
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
            # New metrics
            'Current_ROA': None,
            'FS_ROA': 0,
            'Current_FCFTA': None,
            'FS_FCFTA': 0,
            'ACCRUAL': None,
            'FS_ACCRUAL': 0,
            'Data_Source': 'None',
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
        'ACCRUAL'                    # Higher is better
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

# Example usage
if __name__ == "__main__":
    # Full list of tickers as in original code
    tickers = [
        "AAL", "AAP", "AAPL", "ABBV", "ABEV", "ABNB", "ABT",
        "ACN", "ADBE", "ADGO", "ADI", "ADP", "AEM", "AMAT",
        "AMD", "AMGN", "AMX", "AMZN", "ANF", "ARCO", "ARM", "ASR",
        "AVGO", "AVY", "AZN","ASML","AI","TEAM","CLS","CEG","DECK","RGTI",
        "B", "BA", "BABA","NOW","VST","VRTX","PATH","PDD","XPEV",
         "BAK", "BB", "BHP","BRK-B",
        "BIDU", "BIIB", "BIOX", "BITF", "BKNG", "BKR", "BMY",
        "BP", "BRFS", "BRK-B", "CAAP", "CAH",
         "CAR", "CAT", "CCL", "CDE", "CL", "COIN", "COST", "CRM",
         "CSCO", "CSNA3.SA", "CVS", "CVX", "CX", "DAL", "DD",
        "DE", "DEO", "DHR", "DIS", "DOCU", "DOW",
        "E", "EA", "EBAY", "EBR","EFX", "EQNR", "ERIC",
        "ERJ", "ETSY", "F", "FCX", "FDX",
        "FMX", "FSLR", "GE", "GFI", "GGB", "GILD",
        "GLOB", "GLW", "GM", "GOOGL", "GRMN",
        "GSK", "GT", "HAL", "HAPV3.SA", "HD", "HL", "HMC", "HMY",
        "HOG", "HON", "HPQ", "HSY", "HUT", "HWM",
         "IBM", "INFY", "INTC", "IP",
        "ISRG","JBSS3.SA",
        "JD", "JMIA", "JNJ","JOYY", "KEP", "KGC",
        "KMB", "KO", "KOD", "LAC", "LAR", "LLY",
        "LMT", "LND", "LRCX", "LREN3.SA", "LVS",  "MA", "MCD",
         "MDLZ", "MDT", "MELI", "META", "MGLU3.SA", "MMC", "MMM",
        "MO", "MOS", "MRK", "MRNA", "MRVL", "MSFT", "MSI", "MSTR",
        "MU", "MUX", "NEM", "NFLX", "NG", "NGG", "NIO",
        "NKE", "NTCO3.SA", "NTES", "NUE", "NVDA", "NVS",
        "NXE", "ORCL", "ORLY", "OXY", "PAAS", "PAC", "PAGS", "PANW", "PBI",
        "PBR", "PCAR", "PEP", "PFE", "PG",
        "PHG", "PINS", "PLTR", "PM", "PRIO3.SA", "PSX", "PYPL", "QCOM",
         "RACE", "RBLX", "RENT3.SA", "RIO", "RIOT", "ROKU", "ROST", "RTX",
         "SAP", "SATL", "SBS", "SBUX", "SCCO", "SDA", "SE",
        "SHEL", "SHOP", "SID", "SLB", "SNA", "SNAP", "SNOW", "SONY", "SPCE",
        "SPGI", "SPOT", "STLA", "STNE", "SYY", "T",
        "TCOM", "TEN", "TGT", "TIMB", "TM",
        "TMO", "TMUS", "TRIP", "TSLA", "TSM", "TTE", "TV", "TWLO",
        "TXN", "UAL", "UBER", "UGP", "UL", "UNH",
         "UNP", "URBN", "V", "VALE",
          "VIST", "VIV", "VOD", "VRSN", "VZ", "WBA", "WB", "WEGE3.SA", "WMT", "X", "XOM", "XP", "XRX",
        "XYZ", "YELP", "ZM"
    ]
    
    # For testing, you can use a smaller subset
    test_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX", "NVDA"]
    
    # Process the tickers - use full list or test list as needed
    # Uncomment the line you want to use
    process_all_tickers(test_tickers)  # Use test list for testing
    # process_all_tickers(tickers)  # Use full list for final analysis