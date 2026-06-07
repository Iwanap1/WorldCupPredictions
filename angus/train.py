from typing import Dict
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


file_path = '/users/angus/cursor/WorldCupPredictions/angus/ranking_results.csv'

ranking_df = pd.read_csv('/users/angus/cursor/WorldCupPredictions/angus/additional_data/fifa_mens_rank.csv')


df = pd.read_csv(file_path)
df = df.dropna()



df['score_diff'] = df['score_1'] - df['score_2']

x_train, x_test, y_train, y_test = train_test_split(df[['ranking_1', 'ranking_2']], df[['score_diff', 'score_1']], test_size=0.2, random_state=42)



g1_model = LinearRegression()
g1_model.fit(x_train, y_train['score_1'])

gd_model = LinearRegression()
gd_model.fit(x_train, y_train['score_diff'])



    
def sim_goals(g1_model, gd_model, rank_1: int, rank_2: int):
    """ simulates the goals for the two teams """
    g1 = g1_model.predict([[rank_1, rank_2]])
    gd = gd_model.predict([[rank_1, rank_2]])
    g2 = g1 - gd

    return np.random.poisson(g1), np.random.poisson(g2)


def fetch_rank(team_name: str, year: int):
    """ fetches the rank of the team for the given year """
    return ranking_df[ranking_df['team'] == team_name][ranking_df['date'] == year]['rank'].values[0]

  
def simulate_match(team_1: str, team_2: str, year: int):
    """ simulates the match between the two teams """
    rank_1 = fetch_rank(team_1, year)
    rank_2 = fetch_rank(team_2, year)
    g1, g2 = sim_goals(g1_model, gd_model, rank_1, rank_2)
    return g1[0], g2[0]

print(simulate_match('England', 'France', 2024))


def sim_100_matches(team_1: str, team_2: str, year: int):
    """ simulates 100 matches between the two teams """
    results = []
    for i in range(100):
        g1, g2 = simulate_match(team_1, team_2, year)
        results.append({'team_1': team_1, 'team_2': team_2, 'year': year, 'g1': g1, 'g2': g2})
    results_df = pd.DataFrame(results)

    

    # count the number of times team 1 wins, draws and loses
    team_1_wins = results_df[results_df['g1'] > results_df['g2']].shape[0]
    print(f'{team_1} wins: {team_1_wins}')
    team_2_wins = results_df[results_df['g2'] > results_df['g1']].shape[0]
    print(f'{team_2} wins: {team_2_wins}')
    draws = results_df[results_df['g1'] == results_df['g2']].shape[0]
    print(f'draws: {draws}')
    result_lib = {}
    for g1, g2 in results_df[['g1', 'g2']].values:
        if (g1, g2) not in result_lib:
            result_lib[(g1, g2)] = 0
        result_lib[(g1, g2)] += 1
    
    return result_lib

result_lib = sim_100_matches('Spain', 'Germany', 2024)
ordered_result_lib = sorted(result_lib.items(), key=lambda x: x[1], reverse=True)
print(ordered_result_lib)
