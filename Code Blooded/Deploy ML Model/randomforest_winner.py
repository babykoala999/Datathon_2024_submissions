from data import encoded_data as df1
from average_strikeRate import result_df as df2
from average_economy import result_df as df3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

df1.rename(columns={'id': 'match_id'}, inplace=True)
merged_df = pd.merge(df1, df2, on='match_id', how='inner')
merged_df = pd.merge(merged_df, df3, on='match_id', how='inner')

print(merged_df)

# input and output
X = merged_df[['home_team','away_team','toss_won', 'decision', 'venue_name','away_avg_strike_rate','home_avg_strike_rate','home_avg_economy_rate', 'away_avg_economy_rate']]
y = merged_df['winner']

# Assuming X and y are your feature and target variables
# Specify a test_size that is smaller than the number of samples in your dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the resulting training and testing sets to verify
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42,max_features = None)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
