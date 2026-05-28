import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from model import Trainer, WorldCupScorePredictor

OUTDIR = "models/1"

BASE_CONFIG = {
    "model": {
        "hidden_dim": [16],
        "activation": "ReLU",
        "exclude_cols": ["year", "is_knockout", "team_1_name", "team_2_name"],
        "target_cols": ["team_1_score", "team_2_score"]
    },
    "training": {
        "test_year": 2014,
        "loss": {
            "core_loss_fn": "MSELoss",
            "goal_difference_weight": 0.2,
        },
        "optimiser": {

        }
    }
}

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(exist_ok=False, parents=True)

    df = pd.read_csv("preprocessed_data/dataset.csv")
    train_df = df[~(df["year"] == BASE_CONFIG["training"]["test_year"])]
    test_df = df[df["year"] == BASE_CONFIG["training"]["test_year"]]

    feature_cols = list(train_df.columns) - BASE_CONFIG["model"]["exclude_cols"] - BASE_CONFIG["model"]["target_cols"]
    BASE_CONFIG["model"]["input_dims"] = len(feature_cols)
    BASE_CONFIG["model"]["features"] = feature_cols

    x_train = train_df[feature_cols]
    y_train = train_df[BASE_CONFIG["model"]["target_cols"]]
    x_test = test_df[feature_cols]
    y_test = test_df[BASE_CONFIG["model"]["target_cols"]]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    with open(outdir / "scalers.pkl", "wb") as f:
        pickle.dump(scaler, f)

    trainer = Trainer()
    


if __name__ == "__main__":
    main()