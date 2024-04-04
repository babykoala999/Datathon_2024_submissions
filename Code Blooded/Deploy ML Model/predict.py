from randomforest_winner import classifier
from current_average import team_avg
from data import venue_mapping
import pandas as pd
from sklearn.metrics import accuracy_score

from data import encoded_data as df1
from average_strikeRate import result_df as df2
from average_economy import result_df as df3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

team_mapping = {'GT': 1, 'CSK': 2, 'LSG': 3, 'MI': 4, 'RR': 5, 'RCB': 6, 'KKR': 7, 'PBKS': 8, 'DC': 9, 'SRH': 10}

reverse_team_mapping = {v: k for k, v in team_mapping.items()}

df1.rename(columns={'id': 'match_id'}, inplace=True)
merged_df = pd.merge(df1, df2, on='match_id', how='inner')
merged_df = pd.merge(merged_df, df3, on='match_id', how='inner')

# input and output
X = merged_df[['home_team','away_team','toss_won', 'decision', 'venue_name','away_avg_strike_rate','home_avg_strike_rate','home_avg_economy_rate', 'away_avg_economy_rate']]
y = merged_df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42,max_features = None,max_samples=None)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define the predict function
def predictWinner(home_team, away_team, toss_winner, toss_decision, venue):
    # Get the numeric identifiers for home and away teams
    home_team_id = team_mapping.get(home_team)
    away_team_id = team_mapping.get(away_team)
    
    # Get the average strike rate and economy rate for home and away teams
    home_avg_strike_rate = team_avg.loc[team_avg['Team'] == home_team_id, 'strikeRate'].values[0]
    away_avg_strike_rate = team_avg.loc[team_avg['Team'] == away_team_id, 'strikeRate'].values[0]
    home_avg_economy_rate = team_avg.loc[team_avg['Team'] == home_team_id, 'economyRate'].values[0]
    away_avg_economy_rate = team_avg.loc[team_avg['Team'] == away_team_id, 'economyRate'].values[0]
    
    # Convert toss decision to binary (0 for bat, 1 for field)
    toss_decision_binary = 0 if toss_decision.lower() == 'bat' else 1

    venue_id = venue_mapping.get(venue)

    input_features = [[home_team_id, away_team_id, team_mapping.get(toss_winner), toss_decision_binary, venue_id,away_avg_strike_rate, home_avg_strike_rate, home_avg_economy_rate, away_avg_economy_rate]]
    
    # Make prediction using the trained classifier
    predicted_winner_id = classifier.predict(input_features)[0]
    
    predicted_winner = reverse_team_mapping.get(predicted_winner_id)

    return predicted_winner

# Example usage:
home_team = 'MI'
away_team = 'CSK'
toss_winner = 'MI'
toss_decision = 'bat'
venue = 'Diamond Oval, Kimberley'

winner_prediction = predictWinner(home_team, away_team, toss_winner, toss_decision, venue)

print("Predicted winner:", winner_prediction)