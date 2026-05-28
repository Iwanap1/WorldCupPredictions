from typing import List, Dict
import pandas as pd
from model import WorldCupScorePredictor
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler


class IwanModel:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.model = WorldCupScorePredictor(model_dir)

        with open(model_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        

    def predict(fixtures_to_predict: pd.DataFrame) -> Dict[str, Dict[str, int]]:
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

