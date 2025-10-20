import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t, norm

# Read parameters from Excel file
df_params = pd.read_excel('data.xlsx', sheet_name='info', header=None)
params = dict(zip(df_params[0], df_params[1]))

initial_amount = float(params['Initial amount'])
years = int(params['Simulation period of years'])
expected_return = params['Expected return']   # Convert percentage to decimal
volatility = params['Volatility'] 
df_degrees = int(params['df'])  # degrees of freedom for fat-tailed t-distribution, convert to int
inflation_mean = params['Inflation mean']   # Convert percentage to decimal
inflation_vol = params['Inflation vol']  # Convert percentage to decimal
num_simulations = 10000

# Read cashflow data
df_cashflow = pd.read_excel('data.xlsx', sheet_name='cashflow')

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

# Function to simulate one path
def simulate_path(initial, years, mu, sigma, df, inflation_mu, inflation_sigma, cf_schedule):
    amounts = [initial]
    # Simulate inflation path
    inflation_returns = norm.rvs(loc=inflation_mu, scale=inflation_sigma, size=years)
    inflation_factors = [1.0]  # Year 0
    cum_inf = 1.0
    for inf_ret in inflation_returns:
        cum_inf *= (1 + inf_ret)
        inflation_factors.append(cum_inf)
    
    for y in range(1, years + 1):
        # Generate annual return
        ret = t.rvs(df, loc=mu, scale=sigma)
        # Calculate amount after return
        new_amount = amounts[-1] * (1 + ret)
        # Add cashflows for this year (inflated)
        new_amount += cf_schedule[y] * inflation_factors[y]
        amounts.append(new_amount)
    return amounts

# Run simulations
all_paths = [simulate_path(initial_amount, years, expected_return, volatility, df_degrees, inflation_mean, inflation_vol, cf_per_year) for _ in range(num_simulations)]

# Calculate percentiles for each year
percentiles_over_time = []
for year in range(years + 1):
    year_amounts = [path[year] for path in all_paths]
    percentiles_over_time.append(np.percentile(year_amounts, [10, 25, 50, 75, 90]))

# Transpose for plotting
percentiles_over_time = np.array(percentiles_over_time).T  # Shape: (5, 11)

# Plot percentiles over time
plt.figure(figsize=(10, 6))
years_range = list(range(years + 1))
plt.plot(years_range, percentiles_over_time[0], label='10th percentile', color='red', linestyle='--')
plt.plot(years_range, percentiles_over_time[1], label='25th percentile', color='orange', linestyle='--')
plt.plot(years_range, percentiles_over_time[2], label='50th percentile', color='green')
plt.plot(years_range, percentiles_over_time[3], label='75th percentile', color='blue', linestyle='--')
plt.plot(years_range, percentiles_over_time[4], label='90th percentile', color='purple', linestyle='--')
plt.title('Evolution of Capital: Percentiles Over 10 Years')
plt.xlabel('Year')
plt.ylabel('Capital Amount')
plt.legend()
plt.grid(True)
plt.show()

# Print final percentiles
final_percentiles = percentiles_over_time[:, -1]
print("Final Percentiles:")
print(f"10th: {final_percentiles[0]:.2f}")
print(f"25th: {final_percentiles[1]:.2f}")
print(f"50th: {final_percentiles[2]:.2f}")
print(f"75th: {final_percentiles[3]:.2f}")
print(f"90th: {final_percentiles[4]:.2f}")