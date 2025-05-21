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
        
        # NEW METRICS FROM THE UPDATED REQUEST
        
        # 1. LEVER = long-term debt (t - 1) / total assets (t - 1) - long-term debt (t) / total assets (t)
        lever = None
        fs_lever = 0  # Default to 0
        
        # Try to use annual data first for long-term debt to total assets ratio
        if len(balance_annual.columns) >= 2:  # Need at least 2 years of data
            current_year = balance_annual.columns[0]
            previous_year = balance_annual.columns[1]
            
            # Get the required values
            lt_debt_current = balance_annual.loc["LongTermDebt"][current_year] if "LongTermDebt" in balance_annual.index else None
            lt_debt_previous = balance_annual.loc["LongTermDebt"][previous_year] if "LongTermDebt" in balance_annual.index else None
            total_assets_current = balance_annual.loc["TotalAssets"][current_year] if "TotalAssets" in balance_annual.index else None
            total_assets_previous = balance_annual.loc["TotalAssets"][previous_year] if "TotalAssets" in balance_annual.index else None
            
            # Calculate LEVER
            if (pd.notna(lt_debt_current) and pd.notna(lt_debt_previous) and 
                pd.notna(total_assets_current) and pd.notna(total_assets_previous) and 
                total_assets_current > 0 and total_assets_previous > 0):
                lever = (lt_debt_previous / total_assets_previous) - (lt_debt_current / total_assets_current)
                fs_lever = 1 if lever > 0 else 0
        
        # 2. LIQUID = current ratio (t) - current ratio (t - 1)
        liquid = None
        fs_liquid = 0  # Default to 0
        
        # Try to use annual data for current ratio calculation
        if len(balance_annual.columns) >= 2:  # Need at least 2 years of data
            current_year = balance_annual.columns[0]
            previous_year = balance_annual.columns[1]
            
            # Get the required values
            current_assets_current = balance_annual.loc["CurrentAssets"][current_year] if "CurrentAssets" in balance_annual.index else None
            current_assets_previous = balance_annual.loc["CurrentAssets"][previous_year] if "CurrentAssets" in balance_annual.index else None
            current_liabilities_current = balance_annual.loc["CurrentLiabilities"][current_year] if "CurrentLiabilities" in balance_annual.index else None
            current_liabilities_previous = balance_annual.loc["CurrentLiabilities"][previous_year] if "CurrentLiabilities" in balance_annual.index else None
            
            # Calculate current ratios
            current_ratio_current = None
            current_ratio_previous = None
            
            if (pd.notna(current_assets_current) and pd.notna(current_liabilities_current) and 
                current_liabilities_current > 0):
                current_ratio_current = current_assets_current / current_liabilities_current
            
            if (pd.notna(current_assets_previous) and pd.notna(current_liabilities_previous) and 
                current_liabilities_previous > 0):
                current_ratio_previous = current_assets_previous / current_liabilities_previous
            
            # Calculate LIQUID
            if pd.notna(current_ratio_current) and pd.notna(current_ratio_previous):
                liquid = current_ratio_current - current_ratio_previous
                fs_liquid = 1 if liquid > 0 else 0
        
        # 3. NEQISS = net equity issuance from t - 1 to t
        neqiss = None
        fs_neqiss = 0  # Default to 0
        
        # Try to get equity issuance from cashflow statement
        if len(cashflow_annual.columns) >= 2:  # Need at least 2 years of data
            current_year = cashflow_annual.columns[0]
            
            # Look for common stock issuance or equity issuance metrics
            stock_issuance = None
            if "CommonStockIssuance" in cashflow_annual.index:
                stock_issuance = cashflow_annual.loc["CommonStockIssuance"][current_year]
            elif "StockBasedCompensation" in cashflow_annual.index:
                stock_issuance = cashflow_annual.loc["StockBasedCompensation"][current_year]
            
            # Repurchases
            stock_repurchase = None
            if "RepurchaseOfCapitalStock" in cashflow_annual.index:
                stock_repurchase = cashflow_annual.loc["RepurchaseOfCapitalStock"][current_year]
            
            # Calculate net equity issuance
            if pd.notna(stock_issuance) or pd.notna(stock_repurchase):
                neqiss = 0
                if pd.notna(stock_issuance):
                    neqiss += stock_issuance
                if pd.notna(stock_repurchase):
                    neqiss -= stock_repurchase
                
                # Flag based on whether net issuance is greater than 0
                fs_neqiss = 1 if neqiss > 0 else 0
        
        # 4. ROA = year-over-year change in ROA
        # Note: This is different from the fs_roa we calculated earlier
        yoy_roa_change = None
        fs_roa_change = 0  # Default to 0
        
        if len(income_annual.columns) >= 2 and len(balance_annual.columns) >= 2:
            current_year = income_annual.columns[0]
            previous_year = income_annual.columns[1]
            
            # Get net income for current and previous year
            net_income_current = income_annual.loc["NetIncome"][current_year] if "NetIncome" in income_annual.index else None
            net_income_previous = income_annual.loc["NetIncome"][previous_year] if "NetIncome" in income_annual.index else None
            
            # Get total assets for current and previous year
            total_assets_current = None
            total_assets_previous = None
            
            if current_year in balance_annual.columns and "TotalAssets" in balance_annual.index:
                total_assets_current = balance_annual.loc["TotalAssets"][current_year]
                
            if previous_year in balance_annual.columns and "TotalAssets" in balance_annual.index:
                total_assets_previous = balance_annual.loc["TotalAssets"][previous_year]
            
            # Calculate ROA for current and previous year
            current_year_roa = None
            previous_year_roa = None
            
            if (pd.notna(net_income_current) and pd.notna(total_assets_current) and 
                total_assets_current > 0):
                current_year_roa = net_income_current / total_assets_current
            
            if (pd.notna(net_income_previous) and pd.notna(total_assets_previous) and 
                total_assets_previous > 0):
                previous_year_roa = net_income_previous / total_assets_previous
            
            # Calculate year-over-year change in ROA
            if pd.notna(current_year_roa) and pd.notna(previous_year_roa):
                yoy_roa_change = current_year_roa - previous_year_roa
                fs_roa_change = 1 if yoy_roa_change > 0 else 0
        
        # 5. FCFTA = year-over-year change in FCFTA 
        # This is different from fs_fcfta calculated earlier
        yoy_fcfta_change = None
        fs_fcfta_change = 0  # Default to 0
        
        if len(cashflow_annual.columns) >= 2 and len(balance_annual.columns) >= 2:
            current_year = cashflow_annual.columns[0]
            previous_year = cashflow_annual.columns[1]
            
            # Get FCF for current and previous year
            fcf_current = cashflow_annual.loc["FreeCashFlow"][current_year] if "FreeCashFlow" in cashflow_annual.index else None
            fcf_previous = cashflow_annual.loc["FreeCashFlow"][previous_year] if "FreeCashFlow" in cashflow_annual.index else None
            
            # Get total assets for current and previous year
            total_assets_current = None
            total_assets_previous = None
            
            if current_year in balance_annual.columns and "TotalAssets" in balance_annual.index:
                total_assets_current = balance_annual.loc["TotalAssets"][current_year]
                
            if previous_year in balance_annual.columns and "TotalAssets" in balance_annual.index:
                total_assets_previous = balance_annual.loc["TotalAssets"][previous_year]
            
            # Calculate FCFTA for current and previous year
            current_year_fcfta = None
            previous_year_fcfta = None
            
            if (pd.notna(fcf_current) and pd.notna(total_assets_current) and 
                total_assets_current > 0):
                current_year_fcfta = fcf_current / total_assets_current
            
            if (pd.notna(fcf_previous) and pd.notna(total_assets_previous) and 
                total_assets_previous > 0):
                previous_year_fcfta = fcf_previous / total_assets_previous
            
            # Calculate year-over-year change in FCFTA
            if pd.notna(current_year_fcfta) and pd.notna(previous_year_fcfta):
                yoy_fcfta_change = current_year_fcfta - previous_year_fcfta
                fs_fcfta_change = 1 if yoy_fcfta_change > 0 else 0
        
        # 6. MARGIN = year-over-year change in gross margin
        margin_change = None
        fs_margin = 0  # Default to 0
        
        if len(income_annual.columns) >= 2:
            current_year = income_annual.columns[0]
            previous_year = income_annual.columns[1]
            
            # Get gross profit and revenue for current and previous year
            gross_profit_current = income_annual.loc["GrossProfit"][current_year] if "GrossProfit" in income_annual.index else None
            gross_profit_previous = income_annual.loc["GrossProfit"][previous_year] if "GrossProfit" in income_annual.index else None
            
            revenue_current = income_annual.loc["TotalRevenue"][current_year] if "TotalRevenue" in income_annual.index else None
            revenue_previous = income_annual.loc["TotalRevenue"][previous_year] if "TotalRevenue" in income_annual.index else None
            
            # Calculate gross margin for current and previous year
            current_gross_margin = None
            previous_gross_margin = None
            
            if (pd.notna(gross_profit_current) and pd.notna(revenue_current) and 
                revenue_current > 0):
                current_gross_margin = gross_profit_current / revenue_current
            
            if (pd.notna(gross_profit_previous) and pd.notna(revenue_previous) and 
                revenue_previous > 0):
                previous_gross_margin = gross_profit_previous / revenue_previous
            
            # Calculate year-over-year change in gross margin
            if pd.notna(current_gross_margin) and pd.notna(previous_gross_margin):
                margin_change = current_gross_margin - previous_gross_margin
                fs_margin = 1 if margin_change > 0 else 0
        
        # 7. TURN = year-over-year change in asset turnover
        turn_change = None
        fs_turn = 0  # Default to 0
        
        if len(income_annual.columns) >= 2 and len(balance_annual.columns) >= 2:
            current_year = income_annual.columns[0]
            previous_year = income_annual.columns[1]
            
            # Get revenue for current and previous year
            revenue_current = income_annual.loc["TotalRevenue"][current_year] if "TotalRevenue" in income_annual.index else None
            revenue_previous = income_annual.loc["TotalRevenue"][previous_year] if "TotalRevenue" in income_annual.index else None
            
            # Get total assets for current and previous year
            total_assets_current = None
            total_assets_previous = None
            
            if current_year in balance_annual.columns and "TotalAssets" in balance_annual.index:
                total_assets_current = balance_annual.loc["TotalAssets"][current_year]
                
            if previous_year in balance_annual.columns and "TotalAssets" in balance_annual.index:
                total_assets_previous = balance_annual.loc["TotalAssets"][previous_year]
            
            # Calculate asset turnover for current and previous year
            current_year_turnover = None
            previous_year_turnover = None
            
            if (pd.notna(revenue_current) and pd.notna(total_assets_current) and 
                total_assets_current > 0):
                current_year_turnover = revenue_current / total_assets_current
            
            if (pd.notna(revenue_previous) and pd.notna(total_assets_previous) and 
                total_assets_previous > 0):
                previous_year_turnover = revenue_previous / total_assets_previous
            
            # Calculate year-over-year change in asset turnover
            if pd.notna(current_year_turnover) and pd.notna(previous_year_turnover):
                turn_change = current_year_turnover - previous_year_turnover
                fs_turn = 1 if turn_change > 0 else 0
        
        # 8. Financial Strength P_FS = Sum of all FS metrics / 10
        # Collect all FS metrics
        fs_metrics = [
            fs_roa,              # From original code
            fs_fcfta,            # From original code
            fs_accrual,          # From original code
            fs_lever,            # New metric
            fs_liquid,           # New metric
            fs_neqiss,           # New metric
            fs_roa_change,       # New metric (year-over-year change in ROA)
            fs_fcfta_change,     # New metric (year-over-year change in FCFTA)
            fs_margin,           # New metric
            fs_turn              # New metric
        ]
        
        # Calculate P_FS
        p_fs = sum(fs_metrics) / 10 if len(fs_metrics) == 10 else None
        
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
        print(f"- LEVER: {lever}, FS_LEVER: {fs_lever}")
        print(f"- LIQUID: {liquid}, FS_LIQUID: {fs_liquid}")
        print(f"- NEQISS: {neqiss}, FS_NEQISS: {fs_neqiss}")
        print(f"- ROA Change: {yoy_roa_change}, FS_ROA_Change: {fs_roa_change}")
        print(f"- FCFTA Change: {yoy_fcfta_change}, FS_FCFTA_Change: {fs_fcfta_change}")
        print(f"- MARGIN Change: {margin_change}, FS_MARGIN: {fs_margin}")
        print(f"- TURN Change: {turn_change}, FS_TURN: {fs_turn}")
        print(f"- P_FS (Financial Strength): {p_fs}")
        print(f"- Data source: {'Quarterly TTM' if len(income_quarterly.columns) >= 4 else 'Annual'}")
        print("-------------------")