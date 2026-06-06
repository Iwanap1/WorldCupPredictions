from .model import WorldCupScorePredictor
import torch
import json
from pathlib import Path
from typing import List

class ModelAnalyser:
    def __init__(self, model: WorldCupScorePredictor):
        self.model = model

    def analyse(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, outdir: Path):
        predictions = self.model(x_tensor).tolist()
        predictions = self.norm_predictions(predictions)
        y_true = y_tensor.tolist()

        n_examples = len(y_true)
        results = {"n_correct_scores": 0, "n_correct_results": 0, "n_correct_gd": 0, "n_examples": n_examples}

        for i, prediction in enumerate(predictions):
            true = y_true[i]
            if self.check_correct_score(prediction, true):
                results["n_correct_scores"] += 1
                results["n_correct_results"] += 1
                results["n_correct_gd"] += 1
                continue
            if self.check_correct_result(prediction, true):
                results["n_correct_results"] += 1
            if self.check_correct_gd(prediction, true):
                results["n_correct_gd"] += 1
        
        results["total_score"] = self.compute_score(results)
        
        with open(outdir / "results.json", "w") as f:
            json.dump(results, f)

        for key, value in results.items():
            if key in ["n_examples", "total_score"]:
                continue
            print(f"{key}: {value} / {results['n_examples']}")
        print("Total Score:", results["total_score"])
        
            

    def norm_predictions(self, predictions: List[float]) -> List[int]:
        results = []
        for t1_score, t2_score in predictions:
            results.append([round(t1_score), round(t2_score)])
        return results
    
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
    
    def compute_score(self, results):
        return results["n_correct_scores"] * 2 + results["n_correct_results"] * 3 + results["n_correct_gd"]