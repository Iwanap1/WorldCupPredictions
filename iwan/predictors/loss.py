import torch.nn as nn
import torch

class Loss:
    def __init__(self, config, model_name):
        self.model_name = model_name
        self.config = config
        loss_config = config["training"]["loss"]
        self.loss_fn = getattr(nn, loss_config["core_loss_fn"])()
        if self.model_name == "WorldCupScorePredictor":
            self.goal_difference_weight = loss_config["goal_difference_weight"]


    def __call__(self, y_true_tensor: torch.Tensor, y_pred_tensor: torch.Tensor) -> torch.Tensor:
        if self.model_name == "WorldCupScorePredictor":
            return self.score_prediction_loss(y_true_tensor, y_pred_tensor) 
        else:
            return self.loss_fn(y_pred_tensor, y_true_tensor)

    def score_prediction_loss(self, y_true_tensor: torch.Tensor, y_pred_tensor: torch.Tensor):
        if self.config["model"]["target_cols"] == ["team_1_score", "team_2_score"]:
            loss_team_1 = self.loss_fn(y_true_tensor[:, 0], y_pred_tensor[:, 0])
            loss_team_2 = self.loss_fn(y_true_tensor[:, 1], y_pred_tensor[:, 1])
            loss_gd = self.loss_fn(self.goal_difference(y_true_tensor), self.goal_difference(y_pred_tensor))
            return loss_team_1 + loss_team_2 + (self.goal_difference_weight * loss_gd)
        
        elif len(self.config["model"]["target_cols"]) == 1:
            return self.loss_fn(y_pred_tensor, y_true_tensor)
        
        else:
            raise NotImplementedError(f'Havent implemented a loss function for target cols: {self.config["model"]["target_cols"]}')


    def goal_difference(self, y_tensor: torch.Tensor):
        return y_tensor[:, 0] - y_tensor[:, 1]


    