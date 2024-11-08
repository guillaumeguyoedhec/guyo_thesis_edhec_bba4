import pandas as pd
from pytickersymbols import PyTickerSymbols

stock_data = PyTickerSymbols()
# Define the target indices
target_indices = {"S&P 500", "NASDAQ 100", "DOW JONES", "S&P 100", "S&P 600"}
all_stocks = stock_data.get_all_stocks()

# Create a list of dictionaries for each stock, storing the ticker and additional details
stock_entries = []
for stock in all_stocks:
    stock_indices = set(stock.get("indices", []))
    matched_indices = stock_indices.intersection(target_indices)
    if matched_indices:
        stock_entries.append({
            "Ticker": stock["symbol"],
            "Country": stock.get("country", ""),
            "Indices": ", ".join(matched_indices),
            "Industries": ", ".join(stock.get("industries", [])),
            "Name": stock.get("name", ""),
        })

df_stocks = pd.DataFrame(stock_entries)
df_stocks.to_csv("df_stocks.csv", index=False)