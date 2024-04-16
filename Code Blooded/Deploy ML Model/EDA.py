import pandas as pd
import numpy as np

data=pd.read_csv("data/all_season_summary.csv")
data=data[['home_team','away_team','toss_won','decision','venue_id','winner','super_over','home_runs','home_wickets','away_runs','away_wickets']]

print("describing")
print(data.describe())

print("info")
data.info()

print("nunique")
print(data.nunique())

print("unique home team names")
print(data['home_team'].unique())

print("unique away team names")
print(data['away_team'].unique())

print("data types")
print(data.dtypes)
