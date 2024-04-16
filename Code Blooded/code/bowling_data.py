import pandas as pd
import numpy as np

# Read bowling data
bowling_df = pd.read_csv("data/all_season_bowling_card.csv")

# Filter data for seasons 2018 to 2023
bowling_df = bowling_df[bowling_df['season'].isin(range(2018, 2024))]

# Select relevant columns
bowling_df = bowling_df[['fullName', 'economyRate']]

# Replace '-' with NaN in 'economyRate' column
bowling_df['economyRate'] = bowling_df['economyRate'].replace('-', np.nan)

# Convert 'economyRate' to float
bowling_df['economyRate'] = bowling_df['economyRate'].astype(float)

# Group by name and calculate average economy rate
result_df = bowling_df.groupby('fullName')['economyRate'].mean().reset_index()

