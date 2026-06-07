from typing import List, Dict
import pandas as pd
from iwan.predictors.goal_probability_predictor import WorldCupGoalsPredictor
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
# from iwan.utils import make_tensor_from_df
import torch
import json


class IwanModel:
    TEAM_MAP = {
        "IR Iran": "Iran",
        "Curaçao": "Curacao",
        "Cabo Verde": "Cape Verde"
    }

    def __init__(self, model_dir: str="models/4_goals_softmax", feature_path: str="iwan/preprocessed_data/2026_featurised_squads.json", temperature=0.8):
        model_dir = Path(model_dir)

        with open(model_dir / "configs.json", "r") as f:
            self.cfg = json.load(f)
        
        self.model = self.load_model(model_dir)
        self.temperature = temperature

        with open(feature_path, "r") as f:
            self.squad_features = json.load(f)

        with open(model_dir / "scalers.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        

    def predict(self, fixtures_to_predict: pd.DataFrame) -> Dict[str, Dict[str, int]]:
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
        results = {}
        for _, row in fixtures_to_predict.iterrows():
            team1 = row["team_1_name"]
            team2 = row["team_2_name"]
            isknockout = 0
            res = self.predict_goals(team1, team2, isknockout)
            results[row["id"]] = {
                "team_1_score": int(res[0]),
                "team_2_score": int(res[1])
            }
        return results


    def load_model(self, model_dir):
        model = WorldCupGoalsPredictor(self.cfg)
        state_dict = torch.load(model_dir / "model.pt")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    

    def featurise(self, team1: str, team2: str, feature_order_list: List[str], isknockout: int=0) -> List[float]:
        team1_name = self.TEAM_MAP.get(team1, team1)
        team2_name = self.TEAM_MAP.get(team2, team2)
        team_1_features = self.squad_features[team1_name]
        team_2_features = self.squad_features[team2_name]
        features = {}
        for f in feature_order_list:
            if f == "is_knockout":
                features[f] = float(isknockout)
                continue

            if "team_1" in f:
                features[f] = float(team_1_features[f.replace("team_1_", "")])
            elif "team_2" in f:
                features[f] = float(team_2_features[f.replace("team_2_", "")])
            else:
                raise NotImplementedError(f"featurise function needs to be updated to handle feature {f}")

        return features
    
    def predict_goals(self, team1, team2, isknockout=0):
        team_1 = self.featurise(team1, team2, self.cfg["model"]["features"], isknockout)
        team_2 = self.featurise(team2, team1, self.cfg["model"]["features"], isknockout)
        data = pd.DataFrame([team_1, team_2])
        data_scaled = self.scaler.transform(data)
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
        with torch.no_grad():
            res = self.model.predict_with_temperature(data_tensor, temperature=self.temperature)
        return res