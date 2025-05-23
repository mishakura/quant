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
        
        # Get annual financial data for the last 8 years
        income_annual = ticker.get_income_stmt(freq="yearly")
        balance_annual = ticker.get_balance_sheet(freq="yearly")
        cashflow_annual = ticker.get_cashflow(freq="yearly")
        
        # Get TTM income statement for Operating Income (for EBIT/EV only)
        income_ttm = ticker.get_income_stmt(freq="trailing")
        info = ticker.info

        # Get actual price and mean price target
        actual_price = info.get('regularMarketPrice', None)
        target_mean_price = info.get('targetMeanPrice', None)
        price_target_pct = None
        if actual_price and target_mean_price and actual_price > 0:
            price_target_pct = (target_mean_price / actual_price) - 1

        # Get financial statement currency (no longer needed in CSV)
        financial_currency = info.get('financialCurrency', None)

        # EBIT/EV (Operating Income TTM / Most recent EV)
        operating_income = None
        if 'OperatingIncome' in income_ttm.index:
            operating_income = income_ttm.loc["OperatingIncome"].iloc[0]
        enterprise_value = info.get('enterpriseValue', None)
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
                if year_col in balance_annual.columns:
                    net_income = income_annual.loc["NetIncome"][year_col] if "NetIncome" in income_annual.index else None
                    total_assets = balance_annual.loc["TotalAssets"][year_col] if "TotalAssets" in balance_annual.index else None
                    if pd.notna(net_income) and pd.notna(total_assets) and total_assets != 0:
                        roa_values.append(net_income / total_assets)
        eight_year_roa = None
        if len(roa_values) >= 3:
            roa_for_calc = [(1 + r) for r in roa_values]
            eight_year_roa = np.prod(np.array(roa_for_calc)) ** (1 / len(roa_values)) - 1

        # 2. Eight-Year Return on Capital (Geometric Average)
        roc_values = []
        for i in range(years):
            if i < len(income_annual.columns) and i < len(balance_annual.columns):
                year_col = income_annual.columns[i]
                if year_col in balance_annual.columns:
                    operating_income_annual = income_annual.loc["OperatingIncome"][year_col] if "OperatingIncome" in income_annual.index else None
                    current_assets = balance_annual.loc["CurrentAssets"][year_col] if "CurrentAssets" in balance_annual.index else None
                    current_liabilities = balance_annual.loc["CurrentLiabilities"][year_col] if "CurrentLiabilities" in balance_annual.index else None
                    net_ppe = balance_annual.loc["NetPPE"][year_col] if "NetPPE" in balance_annual.index else None
                    if (
                        pd.notna(operating_income_annual)
                        and pd.notna(current_assets)
                        and pd.notna(current_liabilities)
                        and pd.notna(net_ppe)
                    ):
                        capital = (current_assets - current_liabilities) + net_ppe
                        if capital > 0:
                            roc_values.append(operating_income_annual / capital)
        eight_year_roc = None
        if len(roc_values) >= 3:
            roc_for_calc = [(1 + r) for r in roc_values]
            eight_year_roc = np.prod(np.array(roc_for_calc)) ** (1 / len(roc_values)) - 1

        # 3. Sum (eight-year FCF) / total assets (t)
        fcf_sum = 0
        current_total_assets = None
        if "FreeCashFlow" in cashflow_annual.index and len(cashflow_annual.columns) > 0:
            fcf_values = [cashflow_annual.loc["FreeCashFlow"][year] for year in cashflow_annual.columns[:years]
                         if pd.notna(cashflow_annual.loc["FreeCashFlow"][year])]
            fcf_sum = sum(fcf_values)
        if "TotalAssets" in balance_annual.index and len(balance_annual.columns) > 0:
            current_total_assets = balance_annual.loc["TotalAssets"].iloc[0]
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
        margin_stability = None
        if len(gross_margins) >= 3:
            avg_gross_margin = np.mean(gross_margins)
            std_gross_margin = np.std(gross_margins, ddof=1)
            if std_gross_margin > 0:
                margin_stability = avg_gross_margin / std_gross_margin
        gross_margin_growth = None
        if len(gross_margins) > 1:
            # Only calculate if both are positive and denominator is not zero
            if gross_margins[0] > 0 and gross_margins[-1] > 0:
                gross_margin_growth = (gross_margins[0] / gross_margins[-1]) ** (1 / (len(gross_margins) - 1)) - 1
            else:
                gross_margin_growth = None
        if isinstance(gross_margin_growth, complex):
            gross_margin_growth = float('nan')

        # NEW METRIC 1: Current ROA = Net income before extraordinary items (t) / total assets (t)
        current_roa = None
        fs_roa = 0
        if len(income_annual.columns) > 0 and len(balance_annual.columns) > 0:
            most_recent_year = income_annual.columns[0]
            net_income_current = income_annual.loc["NetIncome"][most_recent_year] if "NetIncome" in income_annual.index else None
            if most_recent_year in balance_annual.columns:
                total_assets_current = balance_annual.loc["TotalAssets"][most_recent_year] if "TotalAssets" in balance_annual.index else None
                if pd.notna(net_income_current) and pd.notna(total_assets_current) and total_assets_current != 0:
                    current_roa = net_income_current / total_assets_current
                    fs_roa = 1 if current_roa > 0 else 0

        # NEW METRIC 2: Current FCFTA = free cash flow (t) / total assets (t)
        current_fcfta = None
        fs_fcfta = 0
        if len(cashflow_annual.columns) > 0 and len(balance_annual.columns) > 0:
            most_recent_year = cashflow_annual.columns[0]
            fcf_current = cashflow_annual.loc["FreeCashFlow"][most_recent_year] if "FreeCashFlow" in cashflow_annual.index else None
            if most_recent_year in balance_annual.columns:
                total_assets_current = balance_annual.loc["TotalAssets"][most_recent_year] if "TotalAssets" in balance_annual.index else None
            else:
                total_assets_current = balance_annual.loc["TotalAssets"].iloc[0] if "TotalAssets" in balance_annual.index else None
            if pd.notna(fcf_current) and pd.notna(total_assets_current) and total_assets_current != 0:
                current_fcfta = fcf_current / total_assets_current
                fs_fcfta = 1 if current_fcfta > 0 else 0

        # NEW METRIC 3: ACCRUAL = FCFTA - ROA
        accrual = None
        fs_accrual = 0
        if pd.notna(current_fcfta) and pd.notna(current_roa):
            accrual = current_fcfta - current_roa
            fs_accrual = 1 if accrual > 0 else 0

        # NEW METRIC 4: LEVER = long-term debt (t − 1) / total assets (t − 1) − long-term debt (t) / total assets (t)
        lever = None
        fs_lever = 0
        if len(balance_annual.columns) >= 2:
            current_year = balance_annual.columns[0]
            previous_year = balance_annual.columns[1]
            lt_debt_current = balance_annual.loc["LongTermDebt"][current_year] if "LongTermDebt" in balance_annual.index else None
            total_assets_current = balance_annual.loc["TotalAssets"][current_year] if "TotalAssets" in balance_annual.index else None
            lt_debt_previous = balance_annual.loc["LongTermDebt"][previous_year] if "LongTermDebt" in balance_annual.index else None
            total_assets_previous = balance_annual.loc["TotalAssets"][previous_year] if "TotalAssets" in balance_annual.index else None
            if (pd.notna(lt_debt_current) and pd.notna(total_assets_current) and 
                pd.notna(lt_debt_previous) and pd.notna(total_assets_previous) and
                total_assets_current > 0 and total_assets_previous > 0):
                lt_debt_to_assets_current = lt_debt_current / total_assets_current
                lt_debt_to_assets_previous = lt_debt_previous / total_assets_previous
                lever = lt_debt_to_assets_previous - lt_debt_to_assets_current
                fs_lever = 1 if lever > 0 else 0

        # NEW METRIC 5: LIQUID = current ratio (t) − current ratio (t − 1)
        liquid = None
        fs_liquid = 0
        if len(balance_annual.columns) >= 2:
            current_year = balance_annual.columns[0]
            previous_year = balance_annual.columns[1]
            current_assets_current = balance_annual.loc["CurrentAssets"][current_year] if "CurrentAssets" in balance_annual.index else None
            current_liab_current = balance_annual.loc["CurrentLiabilities"][current_year] if "CurrentLiabilities" in balance_annual.index else None
            current_assets_previous = balance_annual.loc["CurrentAssets"][previous_year] if "CurrentAssets" in balance_annual.index else None
            current_liab_previous = balance_annual.loc["CurrentLiabilities"][previous_year] if "CurrentLiabilities" in balance_annual.index else None
            if (pd.notna(current_assets_current) and pd.notna(current_liab_current) and 
                pd.notna(current_assets_previous) and pd.notna(current_liab_previous) and
                current_liab_current > 0 and current_liab_previous > 0):
                current_ratio_current = current_assets_current / current_liab_current
                current_ratio_previous = current_assets_previous / current_liab_previous
                liquid = current_ratio_current - current_ratio_previous
                fs_liquid = 1 if liquid > 0 else 0

        # NEW METRIC 6: ROA_Change = year-over-year change in ROA using annual data
        roa_change = None
        fs_roa_change = 0
        if len(income_annual.columns) >= 2 and len(balance_annual.columns) >= 2:
            current_year = income_annual.columns[0]
            previous_year = income_annual.columns[1]
            current_net_income = income_annual.loc["NetIncome"][current_year] if "NetIncome" in income_annual.index else None
            current_total_assets = balance_annual.loc["TotalAssets"][current_year] if "TotalAssets" in balance_annual.index and current_year in balance_annual.columns else None
            previous_net_income = income_annual.loc["NetIncome"][previous_year] if "NetIncome" in income_annual.index else None
            previous_total_assets = balance_annual.loc["TotalAssets"][previous_year] if "TotalAssets" in balance_annual.index and previous_year in balance_annual.columns else None
            current_roa_annual = None
            previous_roa_annual = None
            if pd.notna(current_net_income) and pd.notna(current_total_assets) and current_total_assets > 0:
                current_roa_annual = current_net_income / current_total_assets
            if pd.notna(previous_net_income) and pd.notna(previous_total_assets) and previous_total_assets > 0:
                previous_roa_annual = previous_net_income / previous_total_assets
            if pd.notna(current_roa_annual) and pd.notna(previous_roa_annual):
                roa_change = current_roa_annual - previous_roa_annual
                fs_roa_change = 1 if roa_change > 0 else 0

        # NEW METRIC 7: FCFTA_Change = year-over-year change in FCFTA using annual data
        fcfta_change = None
        fs_fcfta_change = 0
        if len(cashflow_annual.columns) >= 2 and len(balance_annual.columns) >= 2:
            current_year = cashflow_annual.columns[0]
            previous_year = cashflow_annual.columns[1]
            current_fcf = cashflow_annual.loc["FreeCashFlow"][current_year] if "FreeCashFlow" in cashflow_annual.index else None
            current_assets = balance_annual.loc["TotalAssets"][current_year] if "TotalAssets" in balance_annual.index and current_year in balance_annual.columns else None
            previous_fcf = cashflow_annual.loc["FreeCashFlow"][previous_year] if "FreeCashFlow" in cashflow_annual.index else None
            previous_assets = balance_annual.loc["TotalAssets"][previous_year] if "TotalAssets" in balance_annual.index and previous_year in balance_annual.columns else None
            current_fcfta_annual = None
            previous_fcfta_annual = None
            if pd.notna(current_fcf) and pd.notna(current_assets) and current_assets > 0:
                current_fcfta_annual = current_fcf / current_assets
            if pd.notna(previous_fcf) and pd.notna(previous_assets) and previous_assets > 0:
                previous_fcfta_annual = previous_fcf / previous_assets
            if pd.notna(current_fcfta_annual) and pd.notna(previous_fcfta_annual):
                fcfta_change = current_fcfta_annual - previous_fcfta_annual
                fs_fcfta_change = 1 if fcfta_change > 0 else 0

        # NEW METRIC: MARGIN = year-over-year change in gross margin (annual data)
        margin = None
        fs_margin = 0
        if len(income_annual.columns) >= 2:
            current_year = income_annual.columns[0]
            previous_year = income_annual.columns[1]
            gross_profit_current = income_annual.loc["GrossProfit"][current_year] if "GrossProfit" in income_annual.index else None
            total_revenue_current = income_annual.loc["TotalRevenue"][current_year] if "TotalRevenue" in income_annual.index else None
            gross_profit_previous = income_annual.loc["GrossProfit"][previous_year] if "GrossProfit" in income_annual.index else None
            total_revenue_previous = income_annual.loc["TotalRevenue"][previous_year] if "TotalRevenue" in income_annual.index else None
            if (
                pd.notna(gross_profit_current) and pd.notna(total_revenue_current) and total_revenue_current > 0 and
                pd.notna(gross_profit_previous) and pd.notna(total_revenue_previous) and total_revenue_previous > 0
            ):
                gross_margin_current = gross_profit_current / total_revenue_current
                gross_margin_previous = gross_profit_previous / total_revenue_previous
                margin = gross_margin_current - gross_margin_previous
                fs_margin = 1 if margin > 0 else 0

        # NEW METRIC: TURN = year-over-year change in asset turnover (annual data)
        turn = None
        fs_turn = 0
        if len(income_annual.columns) >= 2 and len(balance_annual.columns) >= 2:
            current_year = income_annual.columns[0]
            previous_year = income_annual.columns[1]
            total_revenue_current = income_annual.loc["TotalRevenue"][current_year] if "TotalRevenue" in income_annual.index else None
            total_assets_current = balance_annual.loc["TotalAssets"][current_year] if "TotalAssets" in balance_annual.index and current_year in balance_annual.columns else None
            total_revenue_previous = income_annual.loc["TotalRevenue"][previous_year] if "TotalRevenue" in income_annual.index else None
            total_assets_previous = balance_annual.loc["TotalAssets"][previous_year] if "TotalAssets" in balance_annual.index and previous_year in balance_annual.columns else None
            if (
                pd.notna(total_revenue_current) and pd.notna(total_assets_current) and total_assets_current > 0 and
                pd.notna(total_revenue_previous) and pd.notna(total_assets_previous) and total_assets_previous > 0
            ):
                asset_turnover_current = total_revenue_current / total_assets_current
                asset_turnover_previous = total_revenue_previous / total_assets_previous
                turn = asset_turnover_current - asset_turnover_previous
                fs_turn = 1 if turn > 0 else 0

        # Add Market Cap from ticker.info
        market_cap = info.get('marketCap', None)

        return {
            'Ticker': ticker_symbol,
            'MarketCap': market_cap,
            'ActualPrice': actual_price,
            'TargetMeanPrice': target_mean_price,
            'TargetMeanPricePct': price_target_pct,
            'FinancialCurrency': financial_currency,  # <-- Always include this!
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
            'MARGIN': margin,
            'FS_MARGIN': fs_margin,
            'TURN': turn,
            'FS_TURN': fs_turn,
        }

    except Exception as e:
        print(f"Error getting data for {ticker_symbol}: {e}")
        return {
            'Ticker': ticker_symbol,
            'MarketCap': None,
            'ActualPrice': None,
            'TargetMeanPrice': None,
            'TargetMeanPricePct': None,
            'FinancialCurrency': None,  # <-- Always include this!
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
            'MARGIN': None,
            'FS_MARGIN': 0,
            'TURN': None,
            'FS_TURN': 0,
        }

def process_all_tickers(tickers):
    results = []
    for ticker_symbol in tickers:
        print(f"Processing {ticker_symbol}...")
        metrics = get_financial_metrics(ticker_symbol)
        results.append(metrics)
        time.sleep(1)
    results_df = pd.DataFrame(results)

    # Only filter if the column exists
    if 'FinancialCurrency' in results_df.columns:
        results_df = results_df[results_df['FinancialCurrency'] == 'USD'].reset_index(drop=True)

    metrics_for_percentile = [
        'OperatingIncome_EV_Ratio',
        'Eight_Year_ROA',
        'Eight_Year_ROC',
        'FCF_Sum_to_Assets',
        'Eight_Year_Gross_Margin_Growth',
        'Margin_Stability',
        'Current_ROA',
        'Current_FCFTA',
        'ACCRUAL',
        'LEVER',
        'LIQUID',
        'ROA_Change',
        'FCFTA_Change',
        'MARGIN',
        'TURN',
    ]

    for metric in metrics_for_percentile:
        df_filtered = results_df.dropna(subset=[metric])
        if not df_filtered.empty:
            metric_values = df_filtered[metric].values
            metric_percentiles = [stats.percentileofscore(metric_values, x) / 100 for x in metric_values]
            ticker_to_percentile = dict(zip(df_filtered['Ticker'].values, metric_percentiles))
            results_df[f'{metric}_Percentile'] = results_df['Ticker'].map(ticker_to_percentile)
        else:
            results_df[f'{metric}_Percentile'] = None

    margin_stability_col = 'Margin_Stability_Percentile'
    margin_growth_col = 'Eight_Year_Gross_Margin_Growth_Percentile'
    if margin_stability_col in results_df.columns and margin_growth_col in results_df.columns:
        results_df['Max_Margin_Metric_Percentile'] = results_df[[margin_stability_col, margin_growth_col]].max(axis=1)
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

    franchise_power_columns = [
        'Eight_Year_ROA_Percentile',
        'Eight_Year_ROC_Percentile',
        'FCF_Sum_to_Assets_Percentile',
        'Max_Margin_Metric_Percentile'
    ]
    def calculate_franchise_power(row):
        values = [row[col] for col in franchise_power_columns if pd.notna(row[col])]
        return np.mean(values) if values else None
    results_df['Franchise_Power_Percentile'] = results_df.apply(calculate_franchise_power, axis=1)

    fs_columns = [
        'FS_ROA',
        'FS_FCFTA',
        'FS_ACCRUAL',
        'FS_LEVER',
        'FS_LIQUID',
        'FS_ROA_Change',
        'FS_FCFTA_Change',
        'FS_MARGIN',
        'FS_TURN',
    ]
    results_df['Financial_Strength_Score'] = results_df[fs_columns].sum(axis=1)

    p_fs_columns = [
        'FS_ROA',
        'FS_FCFTA',
        'FS_ACCRUAL',
        'FS_LEVER',
        'FS_LIQUID',
        'FS_ROA_Change',
        'FS_FCFTA_Change',
        'FS_MARGIN',
        'FS_TURN',
    ]
    results_df['P_FS'] = results_df[p_fs_columns].sum(axis=1) / 9

    results_df['QUALITY'] = 0.5 * results_df['Franchise_Power_Percentile'] + 0.5 * results_df['P_FS']
    results_df['QUANTITATIVE_VALUE'] = results_df['QUALITY'] + 2 * results_df['OperatingIncome_EV_Ratio_Percentile']

    print(results_df.head())
    results_df.to_csv('financial_metrics_extended.csv', index=False)
    return results_df

def calculate_composite_score(results_df, weights=None):
    percentile_cols = [col for col in results_df.columns if col.endswith('_Percentile')]
    if not percentile_cols:
        print("No percentile columns found!")
        return results_df
    if weights is None:
        weights = {col: 1.0/len(percentile_cols) for col in percentile_cols}
    for col in percentile_cols:
        if col not in weights:
            print(f"Warning: No weight provided for {col}, using 0")
            weights[col] = 0
    results_df['Composite_Score'] = 0
    for col in percentile_cols:
        mask = results_df[col].notna()
        results_df.loc[mask, 'Composite_Score'] += results_df.loc[mask, col] * weights[col]
    results_df['Metrics_Used'] = results_df[percentile_cols].count(axis=1)
    denominator = 0
    for col in percentile_cols:
        denominator += weights[col]
    total_weight = sum(weights.values())
    if denominator > 0:
        results_df['Composite_Score'] = results_df['Composite_Score'] * (total_weight / denominator)
    results_df_sorted = results_df.sort_values('Composite_Score', ascending=False)
    return results_df_sorted

if __name__ == "__main__":
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

    test_tickers = ["AAPL"]
    process_all_tickers(test_tickers)