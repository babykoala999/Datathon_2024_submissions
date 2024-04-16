import pandas as pd

# Suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read the data into DataFrame
df = pd.read_csv("data/all_season_bowling_card.csv")
df = df[df['season'].isin(range(2018, 2024))]


# Convert 'economyRate' column to numeric type
df['economyRate'] = pd.to_numeric(df['economyRate'], errors='coerce')

# Create a new DataFrame to store results
result_df = pd.DataFrame(columns=['match_id', 'home_avg_economy_rate', 'away_avg_economy_rate'])

# Iterate through each unique match ID
for match_id in df['match_id'].unique():
    match_data = df[df['match_id'] == match_id]
    
    # Calculate average economy rate for home team
    home_avg_economy_rate = match_data.loc[(match_data['innings_id'] == 1), 'economyRate'].mean()
    
    # Calculate average economy rate for away team
    away_avg_economy_rate = match_data.loc[(match_data['innings_id'] == 2), 'economyRate'].mean()

    # Append to the new DataFrame
    result_df = result_df._append({'match_id': match_id,'home_avg_economy_rate': home_avg_economy_rate,'away_avg_economy_rate': away_avg_economy_rate},ignore_index=True)

# Print the new DataFrame
print(result_df)
