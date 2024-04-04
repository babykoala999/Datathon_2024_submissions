import pandas as pd
from sklearn.preprocessing import StandardScaler

# importing the data
summary_df=pd.read_csv("data/all_season_summary.csv")

summary_data=summary_df.dropna()

# custom mapping 
team_mapping = {'GT': 1, 'CSK': 2, 'LSG': 3, 'MI': 4, 'RR': 5, 'RCB': 6, 'KKR': 7, 'PBKS': 8, 'DC': 9, 'SRH': 10, 'KXIP':8}
teams=['CSK','KKR','GT','SRH','PBKS','RCB','RR','LSG','DC','MI']
venue_mapping = {
    'Narendra Modi Stadium, Motera, Ahmedabad': 1,
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 2,
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 3,
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 4,
    'M.Chinnaswamy Stadium, Bengaluru': 5,
    'MA Chidambaram Stadium, Chepauk, Chennai': 6,
    'Arun Jaitley Stadium, Delhi': 7,
    'Barsapara Cricket Stadium, Guwahati': 8,
    'Eden Gardens, Kolkata': 9,
    'Wankhede Stadium, Mumbai': 10,
    'Sawai Mansingh Stadium, Jaipur': 11,
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 12,
    'Brabourne Stadium, Mumbai': 13,
    'Dr DY Patil Sports Academy, Navi Mumbai': 14,
    'Maharashtra Cricket Association Stadium, Pune': 15,
    'Dubai International Cricket Stadium': 16,
    'Sharjah Cricket Stadium': 17,
    'Sheikh Zayed Stadium, Abu Dhabi': 18,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 19,
    'Holkar Cricket Stadium, Indore': 20,
    'Saurashtra Cricket Association Stadium, Rajkot': 21,
    'Green Park, Kanpur': 22,
    'Shaheed Veer Narayan Singh International Stadium, Raipur': 23,
    'Sardar Patel (Gujarat) Stadium, Motera, Ahmedabad': 24,
    'JSCA International Stadium Complex, Ranchi': 25,
    'Barabati Stadium, Cuttack': 26,
    'Nehru Stadium, Kochi': 27,
    'Dr DY Patil Sports Academy, Mumbai': 28,
    'Vidarbha Cricket Association Stadium, Jamtha, Nagpur': 29,
    'Newlands, Cape Town': 30,
    "St George's Park, Port Elizabeth": 31,
    'Kingsmead, Durban': 32,
    'SuperSport Park, Centurion': 33,
    'Buffalo Park, East London': 34,
    'The Wanderers Stadium, Johannesburg': 35,
    'Diamond Oval, Kimberley': 36,
    'Mangaung Oval, Bloemfontein': 37
}
decision_mapping = {'BOWL FIRST': 0, 'BAT FIRST': 1}

# filtering data
filtered_df = summary_df[summary_df['home_team'].isin(teams) | summary_df['away_team'].isin(teams)]
filtered_df = filtered_df[filtered_df['season'].isin(range(2018, 2024))]

# feature selection
selected_columns = ['id','home_team', 'away_team', 'toss_won', 'decision', 'venue_name', 'winner', 'home_runs', 'home_wickets', 'away_runs', 'away_wickets']
filtered_data = filtered_df[selected_columns]

# Replacing decision values with encoded values
encoded_data = filtered_data.replace({"home_team": team_mapping, "away_team": team_mapping, "winner": team_mapping})
encoded_data['decision'] = encoded_data['decision'].replace(decision_mapping)
encoded_data['venue_name'] = encoded_data['venue_name'].replace(venue_mapping)
encoded_data['toss_won'] = (encoded_data['home_team'] == encoded_data['toss_won']).astype(int)

# scaling the runs
scaler = StandardScaler()
encoded_data[['home_runs', 'away_runs']] = scaler.fit_transform(encoded_data[['home_runs', 'away_runs']])

# Printing the encoded data
print(encoded_data)
#print(encoded_data['decision'].unique())