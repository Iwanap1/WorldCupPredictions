from typing import List, Dict
from model import WorldCupScorePredictor
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler


class IwanModel:
    def __init__(self, model_dir, fixtures_to_predict):

        self.fixtures_to_predict = fixtures_to_predict
        model_dir = Path(model_dir)
        self.model = WorldCupScorePredictor(model_dir)

        with open(model_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        
    def predict() -> Dict[str, Dict[str, int]]:
        """"Must not have any required arguments (put those in the __init__), and must return a dict like below:
        {
            "<match_id>": {
                "team_1_score": 0,
                "team_2_score": 1
            },
            ...
        }

        The match_id is in the data.
        """
