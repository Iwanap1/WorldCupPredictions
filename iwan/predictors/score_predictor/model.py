from typing import Dict, Union, Tuple
import torch.nn as nn
import torch
from pathlib import Path
import json


class WorldCupScorePredictor(nn.Module):
    def __init__(self, config_dict_or_dir_path: Union[Dict, Path]):
        super().__init__()
        if isinstance(config_dict_or_dir_path, dict):
            config = config_dict_or_dir_path
        else:
            with open(config_dict_or_dir_path / "config.json", "r") as f:
                config = json.load(f)

        self.model_config = config["model"]
        act = getattr(nn, self.model_config["activation"])()
        
        x = self.model_config["input_dim"]
        self.layers = []
        for layer_dim in self.model_config["hidden_dims"]:
            self.layers.append(nn.Linear(x, layer_dim))
            x = layer_dim
            self.layers.append(act)
        self.layers.append(nn.Linear(x, len(self.model_config["target_cols"])))
        self.layers.append(nn.Softplus())
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out_tensor = self.mlp(features)
        # out_dict = {}
        # for i, output in enumerate(self.model_config["target_cols"]):
        #     out_dict[output] = out_tensor[:, i]

        # return out_tensor, out_dict
        # print(out_tensor)
        return out_tensor