import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle, json
from pathlib import Path
from predictors import Trainer
from predictors.goal_probability_predictor import WorldCupGoalsPredictor, ModelAnalyser
from utils import make_tensor_from_df, convert_goals_scored_to_one_hot

OUTDIR = "models/6_goals_softmax"

BASE_CONFIG = {
    "model": {
        "hidden_dims": [16],
        "activation": "ReLU",
        "exclude_cols": ["year", "target_team", "opposition", "idx", "goals_scored"],
        "max_goals": 5,
        "poisson_probabilities": False # or false for softmax probabilities
    },
    "training": {
        "epochs": 200,
        "test_year": 2018,
        "loss": {
            "core_loss_fn": "CrossEntropyLoss",
        },
        "dataloaders": {
            "eval": {
                "batch_size": None,
                "shuffle": False,
                "drop_last": False
            },
            "train": {
                "batch_size": 32,
                "shuffle": True,
                "drop_last": False
            }
        },
        "optimiser": {
            "optimiser": "AdamW",
            "lr": 5e-5,
            "weight_decay": 1e-6
        }
    }
}

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv("preprocessed_data/goals_scores.csv")
    df, target_cols = convert_goals_scored_to_one_hot(df, BASE_CONFIG["model"]["max_goals"])
    train_df = df[~(df["year"] == BASE_CONFIG["training"]["test_year"])]
    test_df = df[df["year"] == BASE_CONFIG["training"]["test_year"]]

    BASE_CONFIG["model"]["target_cols"] = target_cols
    feature_cols = [c for c in df.columns if c not in BASE_CONFIG["model"]["target_cols"] and c not in BASE_CONFIG["model"]["exclude_cols"]]
    BASE_CONFIG["model"]["input_dim"] = len(feature_cols)
    BASE_CONFIG["model"]["features"] = feature_cols

    x_train = train_df[feature_cols]
    y_train = train_df[BASE_CONFIG["model"]["target_cols"]]
    x_test = test_df[feature_cols]
    y_test = test_df[BASE_CONFIG["model"]["target_cols"]]

    with open(outdir / "configs.json", "w") as f:
        json.dump(BASE_CONFIG, f, indent=4)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    # print(x_train_scaled)
    x_test_scaled = scaler.transform(x_test)
    with open(outdir / "scalers.pkl", "wb") as f:
        pickle.dump(scaler, f)

    tensors = {}
    tensors["x_train"] = make_tensor_from_df(x_train_scaled)
    tensors["y_train"] = make_tensor_from_df(y_train)
    tensors["x_test"] = make_tensor_from_df(x_test_scaled)
    tensors["y_test"] = make_tensor_from_df(y_test)

    model = WorldCupGoalsPredictor(BASE_CONFIG)
    trainer = Trainer(model, BASE_CONFIG, tensors)
    print(trainer.loss_fn.loss_fn.__class__.__name__)
    trainer.train(outdir)
    analyser = ModelAnalyser()
    analyser.analyse(
        trainer.best_model,
        test_df,
        scaler,
        feature_cols,
        outdir
    )


if __name__ == "__main__":
    main()