import pandas as pd
import numpy as np

batting_df=pd.read_csv("data/all_season_batting_card.csv")

batting_df = batting_df[batting_df['season'].isin(range(2018, 2024))]
batting_df=batting_df[['fullName','strikeRate']]

batting_df['strikeRate'] = batting_df['strikeRate'].replace('-', np.nan)

batting_df['strikeRate'] = batting_df['strikeRate'].astype(float)

result_df = batting_df.groupby('fullName', as_index=False)['strikeRate'].mean()

print(result_df)