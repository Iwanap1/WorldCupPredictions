import torch.nn as nn
import torch

class Loss:
    def __init__(self, config):
        loss_config = config["training"]["loss"]
        self.loss_fn = getattr(nn, loss_config["core_loss_fn"])()
        self.goal_difference_weight = loss_config["goal_difference_weight"]

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        pass
    