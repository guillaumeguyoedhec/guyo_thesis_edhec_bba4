import pandas as pd
import matplotlib.pyplot as plt

weights_markowitz_df = pd.read_csv("weights_markowitz.csv")
weights_black_litterman_df = pd.read_csv("weights_black_litterman.csv")
weights_min_variance_df = pd.read_csv("minimum_variance_portfolio_weights.csv")
weights_threshold_df = pd.read_csv("threshold_portfolio_weights.csv")

# Set the weights for each model
weights_markowitz = 0.7
weights_black_litterman = 0.1
weights_min_variance = 0.1
weights_threshold = 0.1

# Merge all
combined_df = weights_markowitz_df.merge(weights_black_litterman_df, on='Ticker', suffixes=('_markowitz', '_black_litterman'))
combined_df = combined_df.merge(weights_min_variance_df, on='Ticker', suffixes=('', '_min_variance'))
combined_df = combined_df.merge(weights_threshold_df, on='Ticker', suffixes=('', '_threshold'))

# Combined weights
combined_weights = (
    weights_markowitz * combined_df['Optimal Weight'] + 
    weights_black_litterman * combined_df['Optimal Weight'] + 
    weights_min_variance * combined_df['Optimal Weight'] +  # Adjust the column name as needed
    weights_threshold * combined_df['Optimal Weight']  # Adjust if necessary
)

# Normalize the combined weights
normalized_weights = combined_weights / combined_weights.sum()

final_weights_df = pd.DataFrame({'Ticker': combined_df['Ticker'], 'Combined Weight': normalized_weights})
final_weights_df.to_csv("combined_models_portfolio_weights.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(final_weights_df['Ticker'], final_weights_df['Combined Weight'], color='orange')
plt.xlabel('Tickers')
plt.ylabel('Weights')
plt.title('Combined Model Weights')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
