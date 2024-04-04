import pandas as pd
from batting_data import result_df as bat_avg
from bowling_data import result_df as bowl_avg
from teams_2024 import players

# Merge player data with batting and bowling averages
merged_df = pd.merge(players, bat_avg, left_on='Player', right_on='fullName', how='left')
merged_df = pd.merge(merged_df, bowl_avg, left_on='Player', right_on='fullName', how='left')

# Group by team and calculate average strike rate and economy for each team
team_avg = merged_df.groupby('Team').agg({'strikeRate': 'mean', 'economyRate': 'mean'}).reset_index()

# Map team abbreviations to their respective numeric identifiers
team_mapping = {'GT': 1, 'CSK': 2, 'LSG': 3, 'MI': 4, 'RR': 5, 'RCB': 6, 'KKR': 7, 'PBKS': 8, 'DC': 9, 'SRH': 10}
team_avg['Team'] = team_avg['Team'].map(team_mapping)

print(team_avg)
