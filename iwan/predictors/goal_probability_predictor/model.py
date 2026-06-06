from typing import Dict, Union, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path
import json
import numpy as np


class WorldCupGoalsPredictor(nn.Module):
    def __init__(self, config_dict_or_dir_path: Union[Dict, Path]):
        super().__init__()
        if isinstance(config_dict_or_dir_path, dict):
            config = config_dict_or_dir_path
        else:
            with open(config_dict_or_dir_path / "config.json", "r") as f:
                config = json.load(f)

        self.model_config = config["model"]
        act = getattr(nn, self.model_config["activation"])()

        self.max_goals = config["model"]["max_goals"]
        if config["model"]["poisson_probabilities"]:
            self.poisson = True
            self.output_dim = 1
            indeces = [i for i in range(self.max_goals + 1)]
            self.indeces_tensor = torch.Tensor(indeces)
            self.factorial_tensor = self.compute_factorial_tensor()
            self.softplus = nn.Softplus()
        else:
            self.poisson = False
            self.output_dim = self.max_goals + 1
        self.input_dim = self.model_config["input_dim"]
        x = self.input_dim
        self.layers = []
        for layer_dim in self.model_config["hidden_dims"]:
            self.layers.append(nn.Linear(x, layer_dim))
            x = layer_dim
            self.layers.append(act)
        self.layers.append(nn.Linear(x, self.output_dim))
        self.mlp = nn.Sequential(*self.layers)


    def forward(self, x: torch.Tensor, return_full_tensor: bool = True, softmax_output: bool = False, temperature=1):
        z = self.mlp(x)

        if self.poisson:
            z = self.softplus(z)
            e = torch.exp(torch.neg(z))
            r = torch.pow(z, self.indeces_tensor)
            p = (e * r) / self.factorial_tensor
            if softmax_output:
                z = F.softmax(p / temperature, dim=1)
            if return_full_tensor:
                return p
            else:
                return torch.argmax(z, dim=1)
            
        else:
            if softmax_output:
                z = F.softmax(z / temperature, dim=1)
            if return_full_tensor:
                return z
            else:
                return torch.argmax(z, dim=1)
                

    def compute_factorial_tensor(self):
        values = [1]
        for i in range(1, self.max_goals + 1):
            current_index = i
            current_value = 1
            while current_index > 0:
                current_value *= current_index
                current_index -= 1
            values.append(current_value)
        return torch.Tensor(values)
    

    def predict_score(self, team_1_x: torch.Tensor, team_2_x: torch.Tensor, temperature=0):
        with torch.no_grad():
            return {
                "team_1_score": self.predict_with_temperature(team_1_x, temperature),
                "team_2_score": self.predict_with_temperature(team_2_x, temperature)
            }


    def predict_with_temperature(self, x, temperature): 
        if temperature == 0:
            return self(x, return_full_tensor=False)
        
        probs = self(x, softmax_output=True, temperature=temperature)
        res = torch.multinomial(probs, num_samples=1)
        return res[:, 0]