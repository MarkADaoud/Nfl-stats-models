import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

nfl_df = pd.read_csv('nfl_scraper.csv')

nfl_df = nfl_df.dropna()

# Aggregate defensive stats by team and season
team_season_stats = nfl_df.groupby(['Season', 'Team']).agg({
    'Int': 'sum',
    'Sk': 'sum',
}).reset_index()

# Aggregate total stats for each team over all seasons
team_total_stats = team_season_stats.groupby('Team').agg({
    'Int': 'sum',
    'Sk': 'sum',
}).reset_index()

team_total_stats['Num_Seasons'] = team_season_stats.groupby('Team')['Season'].count().values

# Create features and targets for prediction
# Use aggregated stats to predict next season's totals
features = team_total_stats[['Int', 'Sk', 'Num_Seasons']]
targets = {
    'Int': team_total_stats['Int'],
    'Sk': team_total_stats['Sk'],
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
    
    # Predict future totals
    latest_stats = team_total_stats[['Team', 'Int', 'Sk', 'Num_Seasons']]
    
    next_season_predictions = model.predict(latest_stats[['Int', 'Sk', 'Num_Seasons']])
    latest_stats[f'Predicted_Total_{stat}'] = next_season_predictions
    predictions[stat] = latest_stats[['Team', f'Predicted_Total_{stat}']]

# Combine all predictions into a single DataFrame
all_predictions = pd.merge(predictions['Int'], predictions['Sk'], on='Team', how='outer')

all_predictions_sorted = all_predictions.sort_values(by='Team').reset_index(drop=True)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("Predicted Defensive Stats for Next Season:")
print(all_predictions_sorted)
