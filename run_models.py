from typing import List, Union
from iwan import IwanModel
from angus import AngusModel
import pandas as pd
import time, json


def load_models() -> List[Union[IwanModel, AngusModel]]:
    iwan = IwanModel(
        model_dir="iwan/models/1",
        feature_path="iwan/preprocessed_data/2026_featurised_squads.json",
        temperature=0.9
    )
    angus = AngusModel()
    return [iwan, angus]


def main():
    fixtures_df = pd.read_csv("data/2026/fixtures_preprocessed.csv")
    models = load_models()

    predictions = {}
    for model in models:
        predictions[model.__class__.__name__] = model.predict(fixtures_df)

    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

    for fixture_row in fixtures_df.iterrows():
        f_id = fixture_row["id"]

        print(f"\n{fixture_row['team_1_name']} v {fixture_row['team_2_name']}")
        time.sleep(1)
        print('---------------------')
        print_predictions(predictions, f_id)
                

def print_predictions(predictions, f_id):
    for model_name, model_predictions in predictions.items():
        try:
            print(f"{model_name}: {model_predictions[f_id]["team_1_score"]} - {model_predictions[f_id]["team_2_score"]}")
            time.sleep(0.5)
        except KeyError:
            try:
                print(f"{model_name}: {model_predictions[str(f_id)]["team_1_score"]} - {model_predictions[str(f_id)]["team_2_score"]}")
                time.sleep(0.5)
            except KeyError:
                raise KeyError(f"Keys for {model_name} predictions are not correct. Expected to find key '{f_id}' but getting keys: {list(model_predictions.keys())}")


if __name__ == "__main__":
    main()