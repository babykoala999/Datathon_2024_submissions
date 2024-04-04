import pandas as pd
import numpy as np
from randomforest_winner import classifier
from current_average import team_avg
from data import venue_mapping
from sklearn.metrics import accuracy_score
from data import encoded_data as df1
from average_strikeRate import result_df as df2
from average_economy import result_df as df3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
    
    # Get the predicted probabilities for each team
    predicted_probabilities = classifier.predict_proba(input_features)[0]
    
    # Get the team names for visualization
    teams = [reverse_team_mapping.get(team_id) for team_id in classifier.classes_]
    
    # Create a bar plot to visualize the predicted probabilities
    colors = plt.cm.viridis(np.linspace(0, 1, len(teams)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(teams, predicted_probabilities, color=colors)
    plt.xlabel('Teams', fontsize=12)
    plt.ylabel('Predicted Probability', fontsize=12)
    plt.title('Predicted Probabilities for Each Team', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add data labels on top of the bars
    for bar, prob in zip(bars, predicted_probabilities):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{prob:.2f}', 
                ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    # Add a legend for the color gradient
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, label='Normalized Probability', ax=plt.gca())  # Add 'ax=plt.gca()' to specify the current Axes

    plt.tight_layout()
    plt.savefig('plot.png', dpi=300)  # Save the plot as an image file
    plt.show()

    predicted_winner = reverse_team_mapping.get(predicted_winner_id)

    return predicted_winner

# Example usage:
home_team = 'CSK'
away_team = 'RCB'
toss_winner = 'RCB'
toss_decision = 'bat'
venue = 'MA Chidambaram Stadium, Chepauk, Chennai'

winner_prediction = predictWinner(home_team, away_team, toss_winner, toss_decision, venue)

print("Predicted winner:", winner_prediction)


