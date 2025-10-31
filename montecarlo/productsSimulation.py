import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t, norm
import os

# Define a function to calculate maximum drawdown using numpy (vectorized)
def max_drawdown(series):
    if len(series) == 0:
        return 0
    peak = np.maximum.accumulate(series)
    drawdown = (peak - series) / peak
    return np.max(drawdown)

# Folder containing input Excel files
products_folder = 'products'
output_file = 'output.xlsx'

# Prepare Excel writer
with pd.ExcelWriter(output_file) as writer:
    for filename in os.listdir(products_folder):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(products_folder, filename)
            sheet_name = filename.replace('.xlsx', '')
            
            # Read parameters from Excel file
            df_params = pd.read_excel(filepath, sheet_name='info', header=None)
            params = dict(zip(df_params[0], df_params[1]))
            
            initial_amount = float(params['Initial amount'])
            years = int(params['Simulation period of years'])
            expected_return = params['Expected return']   # Convert percentage to decimal
            volatility = params['Volatility'] 
            df_degrees = int(params['df'])  # degrees of freedom for fat-tailed t-distribution, convert to int
            inflation_mean = params['Inflation mean']   # Convert percentage to decimal
            inflation_vol = params['Inflation vol']  # Convert percentage to decimal
            distribution = params['Distribution']  # 'Normal' or 'Fat'
            num_simulations = 5000
            
            # Compute cumulative inflation factors using mean inflation
            cum_inf_factors = [1.0]
            for y in range(1, years + 1):
                cum_inf_factors.append(cum_inf_factors[-1] * (1 + inflation_mean))
            
            # Read cashflow data
            df_cashflow = pd.read_excel(filepath, sheet_name='cashflow')
            
            # Compute total inflation factor for the entire period
            total_inflation_factor = (1 + inflation_mean) ** years
            
            # Collect cashflow data for Excel
            cashflow_data = []
            for cf in df_cashflow.itertuples():
                adjusted = cf.Amount * total_inflation_factor
                cashflow_data.append({
                    'Amount': cf.Amount,
                    'Frequency': cf.Frequency,
                    'Starts': cf.Starts,
                    'Ends': cf.Ends,
                    'Adjusted': adjusted
                })
            df_cf = pd.DataFrame(cashflow_data)
            
            # Precompute cashflows per year
            cf_per_year = [0.0] * (years + 1)
            for cf in df_cashflow.itertuples():
                amount = cf.Amount
                freq = cf.Frequency
                start = cf.Starts
                end = cf.Ends if cf.Ends > 0 else years
                for y in range(max(1, start), min(years + 1, end + 1)):
                    if freq == 'o' and y == start:
                        cf_per_year[y] += amount
                    elif freq == 'y':
                        cf_per_year[y] += amount
                    elif freq == 'q':
                        cf_per_year[y] += amount * 4
                    elif freq == 'm':
                        cf_per_year[y] += amount * 12
            
            # Add this to print cf_per_year for debugging
            print(f"Product: {sheet_name}")
            for y in range(1, years + 1):
                print(f"Year {y}: {cf_per_year[y]}")
            print()  # Blank line for separation
            
            # Function to simulate one path (use normal or t-distribution based on distribution, cap at 0, adjust cashflows for inflation, stop adding cashflows if depleted)
            def simulate_path(initial, years, mu_annual, sigma_annual, df, inflation_mu, inflation_sigma, cf_schedule, distribution, debug=False):
                amounts = [initial]
                investment = [initial]  # Track investment balance separately (excluding cashflows)
                returns = []
                # Simulate inflation path
                inflation_factors = [1.0]  # Year 0
                cum_inf = 1.0
                for y in range(1, years + 1):
                    inf_ret = np.random.normal(inflation_mu, inflation_sigma)
                    cum_inf *= (1 + inf_ret)
                    inflation_factors.append(cum_inf)
                
                if debug:
                    print(f"Inflation factors: {inflation_factors}")
                
                for y in range(1, years + 1):
                    # Generate annual return: normal if 'Normal', else t-distribution
                    if distribution == 'Normal':
                        ret = np.random.normal(mu_annual, sigma_annual)
                    else:
                        ret = t.rvs(df, loc=mu_annual, scale=sigma_annual)
                    returns.append(ret)
                    # Update investment balance (no cashflows added)
                    investment.append(investment[-1] * (1 + ret))
                    # Calculate amount after return
                    new_amount = amounts[-1] * (1 + ret)
                    # Cap at 0
                    new_amount = max(0, new_amount)
                    # Add cashflows only if not depleted
                    if new_amount > 0:
                        adjusted_cf = cf_schedule[y] * inflation_factors[y]
                        new_amount += adjusted_cf
                        if debug:
                            print(f"Year {y}: Nominal CF {cf_schedule[y]}, Inflation Factor {inflation_factors[y]:.4f}, Adjusted CF {adjusted_cf:.2f}, New Amount {new_amount:.2f}")
                    amounts.append(new_amount)
                return amounts, returns, inflation_factors, investment  # Added investment
            
            # Run simulations (enable debug for first path)
            all_paths = []
            for i in range(num_simulations):
                debug_flag = (i == 0)  # Debug only the first simulation
                path = simulate_path(initial_amount, years, expected_return, volatility, df_degrees, inflation_mean, inflation_vol, cf_per_year, distribution, debug=debug_flag)
                all_paths.append(path)
            
            # Calculate percentiles for each year
            percentiles_over_time = []
            for year in range(years + 1):
                year_amounts = [path[0][year] for path in all_paths]  # Fixed: access amounts
                percentiles_over_time.append(np.percentile(year_amounts, [10, 25, 50, 75, 90]))
            
            # Transpose for plotting
            percentiles_over_time = np.array(percentiles_over_time).T  # Shape: (5, 11)
            
            # Plot percentiles over time (commented out to avoid multiple popups)
            # plt.figure(figsize=(10, 6))
            # years_range = list(range(years + 1))
            # plt.plot(years_range, percentiles_over_time[0], label='10th percentile', color='red', linestyle='--')
            # plt.plot(years_range, percentiles_over_time[1], label='25th percentile', color='orange', linestyle='--')
            # plt.plot(years_range, percentiles_over_time[2], label='50th percentile', color='green')
            # plt.plot(years_range, percentiles_over_time[3], label='75th percentile', color='blue', linestyle='--')
            # plt.plot(years_range, percentiles_over_time[4], label='90th percentile', color='purple', linestyle='--')
            # plt.title('Evolution of Capital: Percentiles Over 10 Years')
            # plt.xlabel('Year')
            # plt.ylabel('Capital Amount')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
            
            # New: Histogram of end balances for 95% results (between 2.5th and 97.5th percentiles)
            end_balances = [path[0][-1] for path in all_paths]  # Fixed: access amounts
            lower_bound = np.percentile(end_balances, 2.5)
            upper_bound = np.percentile(end_balances, 97.5)
            filtered_balances = [b for b in end_balances if lower_bound <= b <= upper_bound]
            
            # plt.figure(figsize=(10, 6))
            # plt.hist(filtered_balances, bins=50, edgecolor='black', alpha=0.7)
            # plt.title('Portfolio End Balance Histogram (95% Results)')
            # plt.xlabel('End Balance')
            # plt.ylabel('Frequency')
            # plt.grid(True)
            # plt.show()
            
            # Calculate probability of success (e.g., final amount > 0)
            success_count = sum(1 for path in all_paths if path[0][-1] > 0)  # Fixed: access amounts
            probability_success = (success_count / num_simulations) * 100
            
            # New: Calculate maximum drawdown for each path (excluding cashflows)
            max_drawdowns = [max_drawdown(path[3]) for path in all_paths]  # path[3] is investment
            
            # Convert to negative percentages and cap at -100%
            max_drawdowns = [-min(1, dd) * 100 for dd in max_drawdowns]
            
            # Histogram of maximum drawdowns
            # plt.figure(figsize=(10, 6))
            # plt.hist(max_drawdowns, bins=50, edgecolor='black', alpha=0.7)
            # plt.title('Maximum Drawdown Histogram (Excluding Cashflows)')
            # plt.xlabel('Maximum Drawdown (%)')
            # plt.ylabel('Frequency')
            # plt.grid(True)
            # plt.show()
            
            # Final percentiles
            final_percentiles = percentiles_over_time[:, -1]
            
            # Collect stats data for Excel
            stats_data = {
                'Metric': ['Distribution', 'Final 10th Percentile', 'Final 25th Percentile', 'Final 50th Percentile', 'Final 75th Percentile', 'Final 90th Percentile', 'Probability of Success (%)'],
                'Value': [distribution, final_percentiles[0], final_percentiles[1], final_percentiles[2], final_percentiles[3], final_percentiles[4], probability_success]
            }
            df_stats = pd.DataFrame(stats_data)
            
            # Write cashflow data to sheet
            df_cf.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
            
            # Write stats data below cashflows
            df_stats.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(df_cf) + 2)

print(f"Output written to {output_file}")