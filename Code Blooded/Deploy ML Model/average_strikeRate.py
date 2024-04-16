import pandas as pd

# Suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read the data into DataFrame
df = pd.read_csv("data/all_season_batting_card.csv")
df = df[df['season'].isin(range(2018, 2024))]

# Convert 'strikeRate' column to numeric type
df['strikeRate'] = pd.to_numeric(df['strikeRate'], errors='coerce')

# Create a new DataFrame to store results
result_df = pd.DataFrame(columns=['match_id', 'home_avg_strike_rate', 'away_avg_strike_rate'])

# Iterate through each unique match ID
for match_id in df['match_id'].unique():
    match_data = df[df['match_id'] == match_id]
    
    # Calculate average strike rate for home team
    home_avg_strike_rate = match_data.loc[(match_data['current_innings'] == match_data['home_team']), 'strikeRate'].mean()
    
    # Calculate average strike rate for away team
    away_avg_strike_rate = match_data.loc[(match_data['current_innings'] == match_data['away_team']), 'strikeRate'].mean()

    # Append to the new DataFrame
    result_df = result_df._append({'match_id': match_id,'home_avg_strike_rate': home_avg_strike_rate,'away_avg_strike_rate': away_avg_strike_rate},ignore_index=True)

# Print the new DataFrame
print(result_df)
