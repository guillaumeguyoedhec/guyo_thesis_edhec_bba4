import pandas as pd

file_path = 'metrics.csv'
data = pd.read_csv(file_path)
data.dropna(subset=['Treynor Ratio', "Jensen's Alpha", 'Information Ratio', 'Sector', 'Annual Return', 'PE Ratio'], inplace=True)

# Select the three stocks with the highest annual returns
top_three_returns = data.nlargest(3, 'Annual Return')
remaining_data = data.drop(top_three_returns.index)

# Select the top 2 stocks in each sector based on the Treynor Ratio
top_treynor_per_sector = remaining_data.groupby('Sector').apply(
    lambda df: df.nlargest(2, 'Treynor Ratio')
).reset_index(drop=True)

# Filter to get the best stocks in terms of PE Ratio within this subset
top_pe_per_sector = top_treynor_per_sector.groupby('Sector').apply(
    lambda df: df.nsmallest(2, 'PE Ratio')
).reset_index(drop=True)

# Concatenate the three high-return stocks with the other selected stocks
final_selection = pd.concat([top_three_returns, top_pe_per_sector])

# Ensure Jensen's Alpha, Information Ratio, and Annual Return are positive
filtered_stocks = final_selection[
    (final_selection["Jensen's Alpha"] > 0) &
    (final_selection['Information Ratio'] > 0) &
    (final_selection['Annual Return'] > 0)
]

df_final_without_weights = filtered_stocks[['Ticker', 'Sector', 'Treynor Ratio', "Jensen's Alpha", 'Information Ratio', 'Annual Return', 'PE Ratio']]
print(df_final_without_weights)
df_final_without_weights.to_csv('final_without_weights.csv', index=False)