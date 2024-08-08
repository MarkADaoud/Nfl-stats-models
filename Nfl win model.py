import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

nfl_df = pd.read_csv('nfl_scraper.csv')

nfl_df = nfl_df.dropna()

# Create a 'Win' column based on the 'Result' column
nfl_df['Win'] = nfl_df['Result'].apply(lambda x: 1 if 'W' in x else 0)

# Aggregate wins by team and season
team_season_wins = nfl_df.groupby(['Season', 'Team'])['Win'].sum().reset_index()

team_season_wins['Previous_Wins'] = team_season_wins.groupby('Team')['Win'].shift(1)

team_season_wins = team_season_wins.dropna()

features = team_season_wins[['Previous_Wins']]
target = team_season_wins['Win']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Prepare data for next season predictions
# Use the latest available season's wins to predict the next season's wins
latest_wins = team_season_wins.groupby('Team').tail(1)[['Team', 'Previous_Wins']]
latest_wins = latest_wins.rename(columns={'Previous_Wins': 'Current_Wins'})
latest_features = latest_wins[['Current_Wins']].rename(columns={'Current_Wins': 'Previous_Wins'})
next_season_predictions = model.predict(latest_features)

# Add predictions to the DataFrame
latest_wins['Predicted_Win_Precentage'] = next_season_predictions

latest_wins_sorted = latest_wins.sort_values(by='Team').reset_index(drop=True)

print("Predicted Records for Next Season:")
print(latest_wins_sorted[['Team', 'Predicted_Win_Precentage']])
