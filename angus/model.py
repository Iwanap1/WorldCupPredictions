import pandas as pd
from typing import Dict
import pickle
import numpy as np
import os
import importlib


parent_path = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(parent_path, 'models')


class AngusModel:
    def __init__(self, model_path: str):
        self.model_path = MODEL_PATH if model_path is None else model_path
        self.load_model(self.model_path)
        pass
    
    def get_rankings(self):
        """ gets the rankings dictionary """
        rankings = importlib.import_module('2026_rankings').RANKINGS
        return rankings

    def predict(self, fixtures_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """"Must not have any required arguments besides fixtures_df (put those in the __init__), 
        
        Must return a dict like below:
        {
            "<match_id>": {
                "team_1_score": 0,
                "team_2_score": 1
            },
            ...
        }

        fixtures_df has columns: 
            id, match_number, home_team_id, away_team_id, city_id, stage_id, kickoff_at, match_label, team_1_name, team_2_name

            
        use fixtures_df["id"] as the keys for the results.
        """
        TEAM_MAP = {
        "IR Iran": "Iran",
        "Curaçao": "Curacao",
        "Cabo Verde": "Cape Verde"
        }

        results = {}
        for _, row in fixtures_df.iterrows():
            team1 = row["team_1_name"]
            team2 = row["team_2_name"]
            team1 = TEAM_MAP.get(team1, team1)
            team2 = TEAM_MAP.get(team2, team2)
            g1, g2 = self.simulate_match(team1, team2)
            results[row["id"]] = {
                "team_1_score": int(g1),
                "team_2_score": int(g2)
            }
        return results

    def load_model(self, model_path: str):
        with open(model_path + '/g1_model.pkl', 'rb') as f:
            self.g1_model = pickle.load(f)
        with open(model_path + '/gd_model.pkl', 'rb') as f:
            self.gd_model = pickle.load(f)

    def simulate_match(self, team_1: str, team_2: str):
        RANKINGS = self.get_rankings()
        rank_1 = RANKINGS[team_1]
        rank_2 = RANKINGS[team_2]
        g1, g2 = self.sim_goals(rank_1, rank_2)
        return int(g1), int(g2)

    def sim_goals(self, rank_1: int, rank_2: int):
        g1 = max(0, self.g1_model.predict([[rank_1, rank_2]]))
        gd = self.gd_model.predict([[rank_1, rank_2]])
        g2 = max(0, g1 - gd)
        return np.random.poisson(g1), np.random.poisson(g2)

if __name__ == '__main__':
    model = AngusModel(MODEL_PATH)

  

    


        
