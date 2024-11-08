import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

combined_weights_df = pd.read_csv("combined_models_portfolio_weights.csv")
tickers = combined_weights_df['Ticker'].tolist()
weights = combined_weights_df['Combined Weight'].values
tickers.append('SPY') # Include SPY as the benchmark (for computing IR, etc..)
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)
price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
daily_returns = price_data.pct_change().dropna()
portfolio_daily_returns = daily_returns[tickers[:-1]].dot(weights)  # Exclude SPY from weights as we only want it for benchamrking

# Value-at-Risk (VaR) at 95% and 99%
confidence_levels = [0.95, 0.99]
VaR_values = {conf: np.percentile(portfolio_daily_returns, (1 - conf) * 100) for conf in confidence_levels}

# Conditional Value-at-Risk (CVaR) for each confidence level
CVaR_values = {conf: portfolio_daily_returns[portfolio_daily_returns <= VaR].mean() for conf, VaR in VaR_values.items()}

risk_metrics = []
for conf in confidence_levels:
    print(f"Value-at-Risk (VaR) at {conf*100}% confidence: {VaR_values[conf]:.2%}")
    print(f"Conditional Value-at-Risk (CVaR) at {conf*100}% confidence: {CVaR_values[conf]:.2%}")
    risk_metrics.append({'Metric': f'VaR ({conf*100}%)', 'Value': VaR_values[conf]})
    risk_metrics.append({'Metric': f'CVaR ({conf*100}%)', 'Value': CVaR_values[conf]})

# Tracking Error
benchmark_returns = daily_returns['SPY']  # Now SPY is guaranteed to be in the DataFrame
tracking_error = np.std(portfolio_daily_returns - benchmark_returns)
print(f"Tracking Error: {tracking_error:.4f}")
risk_metrics.append({'Metric': 'Tracking Error', 'Value': tracking_error})

# Beta Calculation
covariance = np.cov(portfolio_daily_returns, benchmark_returns)[0][1]
benchmark_variance = np.var(benchmark_returns)
beta = covariance / benchmark_variance
print(f"Beta: {beta:.4f}")
risk_metrics.append({'Metric': 'Beta', 'Value': beta})

# Information Ratio
expected_portfolio_return = np.mean(portfolio_daily_returns) * 252
expected_benchmark_return = np.mean(benchmark_returns) * 252
excess_returns = portfolio_daily_returns - benchmark_returns
information_ratio = expected_portfolio_return / np.std(excess_returns)
print(f"Information Ratio: {information_ratio:.4f}")
risk_metrics.append({'Metric': 'Information Ratio', 'Value': information_ratio})

# Treynor Ratio
risk_free_rate = 0.04326
treynor_ratio = (expected_portfolio_return - risk_free_rate) / beta
print(f"Treynor Ratio: {treynor_ratio:.4f}")
risk_metrics.append({'Metric': 'Treynor Ratio', 'Value': treynor_ratio})

# Monte Carlo Simulation
num_simulations = 10000
simulated_returns = np.random.normal(expected_portfolio_return / 252, np.std(portfolio_daily_returns), 
                                      (num_simulations, 252))

# Calculate the final portfolio value based on the simulations
initial_investment = 100000  # Assuming this initial investment amount
final_values = initial_investment * (1 + simulated_returns).cumprod(axis=1)[:, -1]

plt.figure(figsize=(10, 6))
plt.hist(final_values, bins=50, alpha=0.75)
plt.axvline(x=np.percentile(final_values, 5), color='red', linestyle='dashed', linewidth=2, label='5th Percentile Value')
plt.axvline(x=np.percentile(final_values, 95), color='green', linestyle='dashed', linewidth=2, label='95th Percentile Value')
plt.title('Monte Carlo Simulation of Final Portfolio Values')
plt.xlabel('Final Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

risk_metrics_df = pd.DataFrame(risk_metrics)
risk_metrics_df.to_csv('risk_metrics_results.csv', index=False)
print(risk_metrics_df)