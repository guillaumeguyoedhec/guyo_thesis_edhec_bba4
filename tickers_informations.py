import pandas as pd
import yfinance as yf
from tickers_fetching_symbols import df_stocks

# Create a DataFrame to store the information
info_columns = ["Ticker", "Sector", "Industry", "Market Cap", "PE Ratio", "Beta", "Dividend Yield", "Revenue", "Net Income"]
df_info = pd.DataFrame(columns=info_columns)

# Retrieve information for each ticker
for ticker in df_stocks["Ticker"]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        df_info = pd.concat([df_info, pd.DataFrame({
            "Ticker": [ticker],
            "Sector": [info.get("sector", "N/A")],
            "Industry": [info.get("industry", "N/A")],
            "Market Cap": [info.get("marketCap", "N/A")],
            "PE Ratio": [info.get("trailingPE", "N/A")],
            "Beta": [info.get("beta", "N/A")],
            "Dividend Yield": [info.get("dividendYield", "N/A")],
            "Revenue": [info.get("totalRevenue", "N/A")],
            "Net Income": [info.get("netIncomeToCommon", "N/A")],
        })], ignore_index=True)
    
    except Exception as e:
        print(f"Error with ticker {ticker}: {e}")

df_info.to_csv("tickers_information.csv", index=False)