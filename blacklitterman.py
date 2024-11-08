import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pypfopt import BlackLittermanModel, risk_models, expected_returns, EfficientFrontier
import matplotlib.pyplot as plt

equity_data = pd.read_csv("final_without_weights.csv")
tickers = equity_data['Ticker'].tolist()
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate market-implied expected returns and covariance matrix
S = risk_models.sample_cov(prices)
market_prior = expected_returns.capm_return(prices)

# Define subjective views with confidence levels
views = {
    "NVDA": 0.55,   # Expected return for NVDA
    "TSLA": -0.04,  # I THINK that TSLA won't perform - Expected return for TSLA
    "NFLX": -0.05,  # I THINK that NFLX won't perform - Expected return for NFLX
}
view_confidences = np.array([1.0, 0.4, 0.5])  # Confidence in views

# Black-Litterman model
bl = BlackLittermanModel(S, pi=market_prior, absolute_views=views, omega="idzorek", view_confidences=view_confidences)
bl_returns = bl.bl_returns()  # Adjusted expected returns
bl_cov = bl.bl_cov()          # Adjusted covariance matrix

# Optimize portfolio for maximum Sharpe ratio using adjusted returns and covariance
ef = EfficientFrontier(bl_returns, bl_cov)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("Black-Litterman Optimal Weights:")
for ticker, weight in cleaned_weights.items():
    print(f"{ticker}: {weight:.4f}")
ef.portfolio_performance(verbose=True)

bl_weights_df = pd.DataFrame(cleaned_weights.items(), columns=['Ticker', 'Optimal Weight'])
bl_weights_df.to_csv("weights_black_litterman.csv", index=False)

# Plot the optimal weights
plt.figure(figsize=(12, 8))
plt.bar(bl_weights_df['Ticker'], bl_weights_df['Optimal Weight'], color='skyblue')
plt.xlabel('Ticker')
plt.ylabel('Optimal Weight')
plt.title('Black-Litterman Optimal Portfolio Weights')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()