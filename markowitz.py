import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models

data = pd.read_csv("final_without_weights.csv")
equity_tickers = data["Ticker"].tolist()
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

# Handle potential errors while fetching data (NA or not available data)
try:
    price_data = yf.download(equity_tickers, start=start_date, end=end_date)["Adj Close"]
except Exception as e:
    print(f"Error fetching data: {e}")
    raise

# Calculate expected returns and sample covariance matrix
mu = expected_returns.mean_historical_return(price_data)
S = risk_models.sample_cov(price_data)

# Efficient Frontier for max Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print("Markowitz Portfolio Weights:")
for ticker, weight in cleaned_weights.items():
    print(f"{ticker}: {weight:.4f}")

# performance metrics
expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)

# Calculate min and max feasible returns
ef_min_return = EfficientFrontier(mu, S)
ef_min_return.min_volatility()
min_return, _, _ = ef_min_return.portfolio_performance()

ef_max_return = EfficientFrontier(mu, S)
ef_max_return.max_sharpe()
max_return, _, _ = ef_max_return.portfolio_performance()

# Generate Efficient Frontier points within min and max returns
target_returns = np.linspace(min_return, max_return, 100)
frontier_volatility = []

for r in target_returns:
    ef_temp = EfficientFrontier(mu, S)
    ef_temp.efficient_return(target_return=r)
    _, volatility, _ = ef_temp.portfolio_performance()
    frontier_volatility.append(volatility)

# Plot Efficient Frontier and portfolio point
plt.plot(frontier_volatility, target_returns, 'b--', label="Efficient Frontier")
plt.scatter(annual_volatility, expected_annual_return, color='red', marker='*', s=100, label='Max Sharpe Portfolio')

# Minimum volatility portfolio
min_volatility = ef_min_return.portfolio_performance()[1]
plt.scatter(min_volatility, min_return, color='green', marker='o', s=100, label='Min Volatility Portfolio')

plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.title("Markowitz Model: Efficient Frontier")
plt.legend()
plt.grid(True)
plt.show()

mrkz_weights_df = pd.DataFrame(cleaned_weights.items(), columns=['Ticker', 'Optimal Weight'])
mrkz_weights_df.to_csv("weights_markowitz.csv", index=False)
print("Weights saved to weights_markowitz.csv")