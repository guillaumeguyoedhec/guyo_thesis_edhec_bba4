import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

portfolio_data = pd.read_csv("combined_models_portfolio_weights.csv")
tickers = portfolio_data['Ticker'].tolist()
weights = portfolio_data['Combined Weight'].values
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
prices.index = prices.index.tz_localize(None) # Ensure the dates are timezone-naive
returns = prices.pct_change().dropna()
fama_french_data = pd.read_csv("F-F_Research_Data_5_Factors_2x3.csv", delimiter=";")

def parse_date(date_str):
    if len(date_str) == 6:
        return pd.to_datetime(date_str, format='%Y%m')
    else:
        return pd.NaT

fama_french_data['Date'] = fama_french_data['Date'].astype(str).apply(parse_date)
fama_french_data.set_index('Date', inplace=True)
fama_french_data = fama_french_data.loc[start_date:end_date]
factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
fama_french_data.index = fama_french_data.index.tz_localize(None)
aligned_returns, aligned_factors = returns.align(fama_french_data[factor_cols], join='inner', axis=0)

# Calculate betas for each ticker based on factor regression
betas = {}
for ticker in tickers:
    Y = aligned_returns[ticker]
    X = aligned_factors
    X = sm.add_constant(X)  # for intercept 
    model = sm.OLS(Y, X).fit()
    betas[ticker] = model.params

betas_df = pd.DataFrame(betas).T
betas_df.to_csv("betas.csv", index=True)

# Calculate portfolio risk based on betas and weights
portfolio_risk = sum(weights[i] * betas[ticker]['Mkt-RF'] for i, ticker in enumerate(tickers))

# 1. Betas by Factor for Each Ticker
plt.figure(figsize=(10, 6))
betas_df.plot(kind='bar', figsize=(12, 8))
plt.title("Betas by Factor for Each Ticker")
plt.xlabel("Tickers")
plt.ylabel("Beta Values")
plt.legend(title="Factors")
plt.tight_layout()
plt.show()

# 2. Portfolio Cumulative Return
portfolio_returns = (aligned_returns * weights).sum(axis=1)
cumulative_returns = (1 + portfolio_returns).cumprod() - 1 
plt.figure(figsize=(10, 6))
cumulative_returns.plot()
plt.title("Portfolio Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid()
plt.tight_layout()
plt.show()

# 3. Heatmap of Betas
plt.figure(figsize=(10, 8))
sns.heatmap(betas_df, annot=True, cmap="coolwarm", cbar_kws={'label': 'Beta Value'})
plt.title("Heatmap of Factor Betas for Each Ticker")
plt.xlabel("Factors")
plt.ylabel("Tickers")
plt.tight_layout()
plt.show()

# Calculate the weighted portfolio returns
portfolio_returns = (aligned_returns * weights).sum(axis=1)
Y = portfolio_returns
X = aligned_factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
X = sm.add_constant(X)  # Adds an intercept term
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

# Plot actual portfolio returns versus predicted portfolio returns from the multi-factor model
plt.figure(figsize=(12, 6))
plt.plot(portfolio_returns.index, portfolio_returns, label='Actual Portfolio Returns', color='blue')
plt.plot(portfolio_returns.index, predictions, label='Predicted Portfolio Returns', color='red', linestyle='--')
plt.title("Multi-Factor Regression of Portfolio Returns")
plt.xlabel("Date")
plt.ylabel("Portfolio Returns")
plt.legend()
plt.tight_layout()
plt.show()