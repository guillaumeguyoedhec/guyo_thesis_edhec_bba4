import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = pd.read_csv("final_without_weights.csv")
tickers = data['Ticker'].tolist()
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years back from now
prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
annual_returns = prices.pct_change().mean() * 252  # Annualizing returns
cov_matrix = prices.pct_change().cov() * 252  # Annualized covariance matrix

# Set thresholds for allocation limits
min_threshold = 0.01  # Minimum weight of 1%
max_threshold = 0.10  # Maximum weight of 10%
num_assets = len(tickers)

# Define the function to maximize the Sharpe ratio
def negative_sharpe(weights):
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - 0.04326) / portfolio_volatility  # Assume a risk-free rate of 3.3%
    return -sharpe_ratio

# Constraints: sum of weights = 1 (no short), and each weight between 1% and 10%
constraints = (
    {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},  # Sum of weights = 1
)
bounds = [(min_threshold, max_threshold) for _ in range(num_assets)]

# Initial guess for weights
initial_guess = np.array([1 / num_assets] * num_assets)
result = minimize(negative_sharpe, initial_guess, bounds=bounds, constraints=constraints)

# Extract the optimized weights
optimized_weights = result.x
portfolio_return = np.dot(optimized_weights, annual_returns)
portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
sharpe_ratio = (portfolio_return - 0.04326) / portfolio_volatility

# Display the optimized weights and performance metrics
print("Optimized Weights:", dict(zip(tickers, optimized_weights)))
print(f"Expected Portfolio Return: {portfolio_return:.2%}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
weights_df = pd.DataFrame({'Ticker': tickers, 'Optimal Weight': optimized_weights})
weights_df.to_csv("threshold_portfolio_weights.csv", index=False)

# Plot the weights for visualization
plt.figure(figsize=(10, 6))
plt.bar(tickers, optimized_weights, color='skyblue')
plt.xlabel('Tickers')
plt.ylabel('Weight')
plt.title('Optimized Portfolio Weights with Thresholds')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()