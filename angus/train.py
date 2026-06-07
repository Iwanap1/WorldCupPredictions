from typing import Dict
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pickle


RESULTS_PATH = '/users/angus/cursor/WorldCupPredictions/angus/ranking_results.csv'

RANKING_PATH = '/users/angus/cursor/WorldCupPredictions/angus/additional_data/fifa_mens_rank.csv'

MODEL_PATH = '/users/angus/cursor/WorldCupPredictions/angus/models/'

def get_training_data(results_path: str):
    results_df = pd.read_csv(results_path)
    results_df = results_df.dropna()
    print(results_df.head())
    results_df['score_diff'] = results_df['score_1'] - results_df['score_2']
    x_train, x_test, y_train, y_test = train_test_split(results_df[['ranking_1', 'ranking_2']], results_df[['score_diff', 'score_1']], test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def train_model(results_path: str):
    x_train, x_test, y_train, y_test = get_training_data(results_path)
    g1_model = LinearRegression()
    g1_model.fit(x_train, y_train['score_1'])
    gd_model = LinearRegression()
    gd_model.fit(x_train, y_train['score_diff'])
    save_model(g1_model, gd_model, MODEL_PATH)
    return g1_model, gd_model

def save_model(g1_model, gd_model, path: str):
    os.makedirs(path, exist_ok=True)
    with open(path + '/g1_model.pkl', 'wb') as f:
        pickle.dump(g1_model, f)
    with open(path + '/gd_model.pkl', 'wb') as f:
        pickle.dump(gd_model, f)

def load_model(path: str):
    with open(path + '/g1_model.pkl', 'rb') as f:
        g1_model = pickle.load(f)
    with open(path + '/gd_model.pkl', 'rb') as f:
        gd_model = pickle.load(f)
    return g1_model, gd_model

def sim_goals(g1_model, gd_model, rank_1: int, rank_2: int):
    g1 = g1_model.predict([[rank_1, rank_2]])
    gd = gd_model.predict([[rank_1, rank_2]])
    g2 = g1 - gd
    return np.random.poisson(g1), np.random.poisson(g2)

def fetch_rank(team_name: str, year: int, ranking_path: str):
    """ fetches the rank of the team for the given year """
    ranking_df = pd.read_csv(ranking_path)
    return ranking_df[ranking_df['team'] == team_name][ranking_df['date'] == year]['rank'].values[0]

def simulate_match(team_1: str, team_2: str, year: int, model_path: str, ranking_path: str):
    # laod models
    rank_1 = fetch_rank(team_1, year, ranking_path)
    rank_2 = fetch_rank(team_2, year, ranking_path)
    with open(model_path + '/g1_model.pkl', 'rb') as f:
        g1_model = pickle.load(f)
    with open(model_path + '/gd_model.pkl', 'rb') as f:
        gd_model = pickle.load(f)
    g1, g2 = sim_goals(g1_model, gd_model, rank_1, rank_2)
    return g1[0], g2[0]

def sim_100_matches(team_1: str, team_2: str, year: int, model_path: str, ranking_path: str):
    """ simulates 100 matches between the two teams """
    results = []
    for i in range(100):
        g1, g2 = simulate_match(team_1, team_2, year, model_path, ranking_path)
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


if __name__ == '__main__':
    train_model(RESULTS_PATH)
    print(simulate_match('Spain', 'Germany', 2024, MODEL_PATH, RANKING_PATH))
    print(sim_100_matches('Spain', 'Germany', 2024, MODEL_PATH, RANKING_PATH))