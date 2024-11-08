import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import datetime as dt
import matplotlib.pyplot as plt

data = pd.read_csv("final_without_weights.csv")
equity_tickers = data['Ticker'].tolist()
start_date = dt.datetime.now() - dt.timedelta(days=365 * 5)  # Last 5 years
end_date = dt.datetime.now()
try:
    price_data_equities = yf.download(equity_tickers, start=start_date, end=end_date)['Adj Close']
except Exception as e:
    print(f"Error fetching data: {e}")
    raise
daily_returns_equities = price_data_equities.pct_change().dropna()
cov_matrix_equities = daily_returns_equities.cov()

# Set a random initial guess to avoid bias in optimization
initial_guess = np.random.dirichlet(np.ones(len(equity_tickers)), size=1)[0]
# Allow some assets to have zero weight by relaxing the lower bound
bounds = [(0.01, 0.2) for _ in range(len(equity_tickers))]
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}) # final weights 1

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))
result = minimize(portfolio_variance, initial_guess, args=(cov_matrix_equities,), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x
print("Minimum Variance Portfolio Weights:")
for ticker, weight in zip(equity_tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

min_variance = portfolio_variance(optimal_weights, cov_matrix_equities)
print("\nMinimum Portfolio Variance:", min_variance)

expected_returns = daily_returns_equities.mean() * 252
expected_portfolio_return = np.dot(optimal_weights, expected_returns)
annual_volatility = np.sqrt(min_variance) * np.sqrt(252)

print(f"\nExpected Annual Return: {expected_portfolio_return:.4f}")
print(f"Annual Volatility: {annual_volatility:.4f}")

weights_df = pd.DataFrame({'Ticker': equity_tickers, 'Optimal Weight': optimal_weights})
weights_df.to_csv("minimum_variance_portfolio_weights.csv", index=False)
print("Weights saved to minimum_variance_portfolio_weights.csv")

plt.figure(figsize=(10, 6))
plt.bar(weights_df['Ticker'], weights_df['Optimal Weight'], color='skyblue')
plt.xlabel('Tickers')
plt.ylabel('Weights')
plt.title('Minimum Variance Portfolio Weights')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()