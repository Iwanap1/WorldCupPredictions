import pandas as pd
from typing import List, Union
from iwan import IwanModel
from angus import AngusModel

def load_models() -> List[Union[IwanModel, AngusModel]]:
    iwan = IwanModel(
        model_dir="iwan/models/6_goals_softmax",
        feature_path="iwan/preprocessed_data/2026_featurised_squads.json",
        temperature=0.45
    )
    angus = AngusModel()
    return [iwan, angus]

def main():
    fixtures_df = pd.read_csv("data/2026/fixtures_preprocessed.csv")
    models = load_models()
    test_df = fixtures_df.sample(20)
    model_predictions = {}
    for model in models:
        model_predictions[model.__class__.__name__] = model.predict(test_df)

    print("Model Names:", list(model_predictions.keys()))

    all_ids = [int(i) for i in test_df["id"].unique().tolist()]


    for model, predictions in model_predictions.items():
        # example_key = list(predictions.keys())[0]
        # print(f"Datatype of key inside {model} predictions:", type(example_key))
        if not isinstance(predictions, dict):
            raise ValueError(f"Predictions for {model} are a {type(predictions)} instead of a Dict")
        # print(predictions.keys())
        for fid, preds in predictions.items():
            if "team_1_score" not in preds:
                raise KeyError(f"team_1_score not in prediction dict for {model}, keys found are: {preds.keys()}")
            if "team_2_score" not in preds:
                raise KeyError(f"team_2_score not in prediction dict for {model}, keys found are: {preds.keys()}")
            if int(fid) not in all_ids:
                raise KeyError(f"Fixture ID in predictions for {model} were not in the DF")
            
        print(f"{model} passed the tests")


if __name__ == "__main__":
    main()