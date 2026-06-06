from pathlib import Path
import os

from dotenv import load_dotenv
import pandas as pd

load_dotenv(Path(__file__).resolve().parents[2] / ".env.local")

DATASET = "lucasyukioimafuko/fifa-mens-world-ranking"
FILENAME = "fifa_mens_rank.csv"
OUT_DIR = Path(__file__).resolve().parent


def get_rankings() -> pd.DataFrame:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(DATASET, FILENAME, path=str(OUT_DIR))
    return pd.read_csv(OUT_DIR / FILENAME)


if __name__ == "__main__":
    df = get_rankings()
    print(df.head())
