import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

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
        
        # 4. Eight-year gross margin growth (geometric average)
        gross_margins = []
        
        for i in range(years):
            if i < len(income_annual.columns):
                year_col = income_annual.columns[i]
                gross_profit = income_annual.loc["GrossProfit"][year_col] if "GrossProfit" in income_annual.index else None
                total_revenue = income_annual.loc["TotalRevenue"][year_col] if "TotalRevenue" in income_annual.index else None
                
                if pd.notna(gross_profit) and pd.notna(total_revenue) and total_revenue != 0:
                    gross_margins.append(gross_profit / total_revenue)
        
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
        
        # Add debug info for this ticker
        print(f"Debug for {ticker_symbol}:")
        print(f"- ROA values: {roa_values}")
        print(f"- ROC values: {roc_values}")
        print(f"- Eight-Year ROA: {eight_year_roa}")
        print(f"- Eight-Year ROC: {eight_year_roc}")
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
            'Years_Data_Available': years,
            'ROA_Data_Points': len(roa_values),
            'ROC_Data_Points': len(roc_values)
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
            'Years_Data_Available': 0,
            'ROA_Data_Points': 0,
            'ROC_Data_Points': 0
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
    
    # Display the results
    print(results_df.head())
    
    # Save to CSV
    results_df.to_csv('financial_metrics_extended.csv', index=False)
    
    return results_df

# List of tickers from your original code
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
test = ["AAPL", "GOOG", "MSFT"]
# Uncomment the line below to run on a small sample first


# Uncomment the line below to run on all tickers
final_df = process_all_tickers(test)

#asd