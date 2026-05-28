from typing import Dict, Union
import torch.nn as nn
from pathlib import Path
import json


class WorldCupScorePredictor(nn.Module):
    def __init__(self, config_dict_or_dir_path: Union[Dict, Path]):

        if isinstance(config_dict_or_dir_path, dict):
            config = config_dict_or_dir_path
        else:
            with open(config_dict_or_dir_path / "config.json", "r") as f:
                config = json.load(f)

        model_config = config["model"]
        act = getattr(nn, model_config["activation"])
        
        x = model_config["input_dims"]
        self.layers = []
        for layer_dim in model_config["hidden_dims"]:
            self.layers.append(nn.Linear(x, layer_dim))
            x = layer_dim
            self.layers.append(act)
        self.layers.append(nn.Linear(x, len(model_config["target_cols"])))
        self.mlp = nn.Sequential(self.layers)

    def forward(self, features):
        return self.mlp(features)