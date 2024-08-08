import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

nfl_df = pd.read_csv('nfl_scraper.csv')

nfl_df = nfl_df.dropna()

# Aggregate offensive stats by team and season
team_season_stats = nfl_df.groupby(['Season', 'Team']).agg({
    'Yds': 'sum',
    'TD': 'sum',
}).reset_index()

# Create features and targets for prediction
# Use previous seasons' stats to predict the next season's stats
team_season_stats['Prev_Yds'] = team_season_stats.groupby('Team')['Yds'].shift(1)
team_season_stats['Prev_TD'] = team_season_stats.groupby('Team')['TD'].shift(1)

team_season_stats = team_season_stats.dropna()

# Define features and targets
features = team_season_stats[['Prev_Yds', 'Prev_TD']]
targets = {
    'Yds': team_season_stats['Yds'],
    'TD': team_season_stats['TD'],
}

models = {}
predictions = {}

# Train models for each statistic
for stat in targets.keys():
    X_train, X_test, y_train, y_test = train_test_split(features, targets[stat], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    models[stat] = model
    y_pred = model.predict(X_test)
    
    # Predict future stats
    latest_stats = team_season_stats.groupby('Team').tail(1)[['Team', f'Prev_{stat}']].rename(columns={f'Prev_{stat}': f'Current_{stat}'})
    
    latest_features = latest_stats[[f'Current_{stat}']].rename(columns={f'Current_{stat}': f'Prev_{stat}'})
    
    latest_features = latest_features.reindex(columns=features.columns, fill_value=0)
    
    next_season_predictions = model.predict(latest_features)
    latest_stats[f'Predicted_Per_Game_{stat}'] = next_season_predictions
    predictions[stat] = latest_stats[['Team', f'Predicted_Per_Game_{stat}']]

# Combine all predictions into a single DataFrame
all_predictions = pd.merge(predictions['Yds'], predictions['TD'], on='Team', how='outer')

all_predictions_sorted = all_predictions.sort_values(by='Team').reset_index(drop=True)

print("Predicted Offensive Stats for Next Season:")
print(all_predictions_sorted)