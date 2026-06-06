import torch
import pandas as pd
from typing import Union, List, Tuple
from numpy import ndarray


def make_tensor_from_df(data: Union[pd.DataFrame, ndarray, List[List[float]]]) -> torch.Tensor:
    if isinstance(data, pd.DataFrame):
        return torch.Tensor(data.values.tolist())
    else:
        return torch.Tensor(data)


def convert_goals_scored_to_one_hot(df: pd.DataFrame, max_goals: int) -> Tuple[pd.DataFrame, List[str]]:
    target_cols = []
    for i in range(max_goals + 1):
        if i != max_goals:
            target_col = f"scored_{i}_goals"
            df.loc[df["goals_scored"] == i, target_col] = 1
            df.loc[~(df["goals_scored"] == i), target_col] = 0
        
        else:
            target_col = f"scored_{i}+_goals"
            df.loc[df["goals_scored"] >= i, target_col] = 1
            df.loc[df["goals_scored"] < i, target_col] = 0
        target_cols.append(target_col)

    return df, target_cols
