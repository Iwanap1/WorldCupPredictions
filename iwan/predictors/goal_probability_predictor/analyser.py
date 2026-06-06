from .model import WorldCupGoalsPredictor
import torch, json
from pathlib import Path
import pandas as pd
from utils import make_tensor_from_df
from sklearn.preprocessing import StandardScaler
from typing import List

class ModelAnalyser:
    def analyse(
        self, 
        model: WorldCupGoalsPredictor, 
        df: pd.DataFrame, 
        scaler: StandardScaler,
        feature_cols: List[str],
        outdir: Path
    ):
        df[feature_cols] = scaler.transform(df[feature_cols])
        match_ids = df["idx"].unique().tolist()
        results = {
            "n_examples": int(len(df) / 2),
            "correct_goal_count": 0,
            "n_correct_scores": 0,
            "n_correct_results": 0,
            "n_correct_gd": 0
        }
        
        for m_id in match_ids:
            subset = df[df["idx"] == m_id]
            teams = subset["target_team"].values
            print("\n", teams[0], "v", teams[1])
            print('---------------------')
            goals = subset["goals_scored"].values.tolist()
            print("Real:", goals[0], "-", goals[1])
            x_tensor = make_tensor_from_df(subset[feature_cols])
            predicted = model.predict_with_temperature(x_tensor, 0).tolist()
            print("Predicted:", predicted[0], "-", predicted[1])

            for i, goal_count in enumerate(goals):
                if int(goal_count) == int(predicted[i]):
                    results["correct_goal_count"] += 1
            if self.check_correct_score(predicted, goals):
                results["n_correct_scores"] += 1
                results["n_correct_results"] += 1
                results["n_correct_gd"] += 1
                continue
            if self.check_correct_result(predicted, goals):
                results["n_correct_results"] += 1
            if self.check_correct_gd(predicted, goals):
                results["n_correct_gd"] += 1
        
        results["total_score"] = self.compute_score(results)
        with open(outdir / "results.json", "w") as f:
            json.dump(results, f)

        for key, value in results.items():
            if key in ["n_examples", "total_score"]:
                continue
            print(f"{key}: {value} / {results['n_examples']}")
        print("Total Score:", results["total_score"])
            

    def compute_score(self, results):
        return results["n_correct_scores"] * 2 + results["n_correct_results"] * 3 + results["n_correct_gd"]

    def check_correct_score(self, predicted_score: List[int], true_score: List[int]):
        if predicted_score[0] == true_score[0] and predicted_score[1] == true_score[1]:
            return True
        return False

    def check_correct_result(self, predicted_score: List[int], true_score: List[int]):
        if predicted_score[0] > predicted_score[1] and true_score[0] > true_score[1]:
            return True
        if predicted_score[0] < predicted_score[1] and true_score[0] < true_score[1]:
            return True
        if predicted_score[0] == predicted_score[1] and true_score[0] == true_score[1]:
            return True
        return False

    def check_correct_gd(self, predicted_score: List[int], true_score: List[int]):
        if predicted_score[0] - predicted_score[1] == true_score[0] - true_score[1]:
            return True
        return False