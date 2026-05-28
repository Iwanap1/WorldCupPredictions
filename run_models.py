from iwan import IwanModel
from angus import AngusModel
import pandas as pd


def load_models():
    iwan = IwanModel("iwan/models/1")
    angus = AngusModel()

    return [iwan, angus]



def main():
    fixtures_df = pd.read_csv("data/2026/fixtures_preprocessed.csv")
    models = load_models()

    predictions = {}
    for model in models:
        predictions[model.__name__] = model.predict(fixtures_df)

    for fixture_row in fixtures_df.iterrows():
        f_id = fixture_row["id"]

        print(f"\n{fixture_row['team_1_name']} v {fixture_row['team_2_name']}")
        print_predictions(predictions, f_id)
                


def print_predictions(predictions, f_id):
    for model_name, model_predictions in predictions.items():
        try:
            print(f"{model_name}: {model_predictions[f_id]["team_1_score"]} - {model_predictions[f_id]["team_2_score"]}")
        except KeyError:
            try:
                print(f"{model_name}: {model_predictions[str(f_id)]["team_1_score"]} - {model_predictions[str(f_id)]["team_2_score"]}")
            except KeyError:
                raise KeyError(f"Keys for {model_name} predictions are not correct. Expected to find key '{f_id}' but getting keys: {list(model_predictions.keys())}")


if __name__ == "__main__":
    main()