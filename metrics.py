import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from tickers_informations import df_info

# tickers, dates, benchmark and returns
tickers = df_info['Ticker'].tolist() + ['^GSPC']  # Add the S&P 500
start_date = "2020-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')
risk_free_rate = 0.04326
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
returns = data.pct_change()
sp500_returns = returns['^GSPC']

# Calculate metrics for each asset
for ticker in tickers[:-1]:  # Exclude the S&P 500 to the pool of ticker
    asset_returns = returns[ticker]
    cov_matrix = returns.cov()
    var_market = sp500_returns.var()

    beta = cov_matrix.loc[ticker, '^GSPC'] / var_market
    alpha = asset_returns.mean() * 252 - (risk_free_rate + beta * (sp500_returns.mean() * 252 - risk_free_rate))
    treynor_ratio = (asset_returns.mean() * 252 - risk_free_rate) / beta
    tracking_error = np.sqrt(((asset_returns - sp500_returns) ** 2).mean()) * np.sqrt(252)
    information_ratio = (asset_returns.mean() - sp500_returns.mean()) / tracking_error

    # Calculate the compound annual return
    cumulative_return = (1 + asset_returns).cumprod()
    annual_return = cumulative_return.iloc[-1]**(252/len(asset_returns)) - 1

    df_info.loc[df_info['Ticker'] == ticker, 'Beta'] = beta
    df_info.loc[df_info['Ticker'] == ticker, 'Treynor Ratio'] = treynor_ratio
    df_info.loc[df_info['Ticker'] == ticker, "Jensen's Alpha"] = alpha
    df_info.loc[df_info['Ticker'] == ticker, 'Tracking Error'] = tracking_error
    df_info.loc[df_info['Ticker'] == ticker, 'Information Ratio'] = information_ratio
    df_info.loc[df_info['Ticker'] == ticker, 'Annual Return'] = annual_return
df_info.to_csv("metrics.csv", index=False)