import numpy as np
import pandas as pd
import random
import time

# Pick range of seasons to scrape data for
seasons = [str(season) for season in range(2010,2023)]
print(f'number of seasons={len(seasons)}')

team_abbrs = [
    'crd','atl','rav','buf','car','chi','cin','cle','dal','den','det','gnb','htx','clt','jax','kan',
    'rai','sdg','ram','mia','min','nwe','nor','nyg','nyj','phi','pit','sfo','sea','tam','oti','was']

print(f'number of teams={len(team_abbrs)}')

nfl_df = pd.DataFrame()

for season in seasons:
  for team in team_abbrs:
    url = 'https://www.pro-football-reference.com/teams/' + team + '/' + season + '/gamelog/'
    print(url)

    # Go through and scrape the stats for each season and team
    off_df = pd.read_html(url, header=1, attrs={'id':'gamelog' + season})[0]

    def_df = pd.read_html(url, header=1, attrs={'id':'gamelog_opp' + season})[0]

    team_df = pd.concat([off_df, def_df], axis=1)

    team_df.insert(loc=0, column='Season', value=season)

    team_df.insert(loc=2, column='Team', value=team.upper())

    nfl_df = pd.concat([nfl_df, team_df], ignore_index=True)

    # Put code to sleep to avoid scrapping problems
    time.sleep(random.randint(7, 8))

nfl_df.to_csv('nfl_scraper.csv', index=False)
