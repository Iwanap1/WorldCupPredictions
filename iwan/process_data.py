from data_processing import SquadParser, FeaturiseSquads, BuildDataset
import json

if __name__ == "__main__":

    parser = SquadParser(
        data_dir="../data/worldcup",
        min_year=1994
    )

    print(f"Parsed squads for years: {list(parser.parsed_squads.keys())}")

    with open("preprocessed_data/new_parsed_squads.json", "w") as f:
        json.dump(parser.parsed_squads, f, indent=4)

    featuriser = FeaturiseSquads(
        parsed_squads=parser.parsed_squads,
        prem_matches_csv_path="raw_data/prem_matches.csv",
        laliga_dir="raw_data/la_liga/"
    )

    with open("preprocessed_data/featurised_squads.json", "w") as f:
        json.dump(featuriser.featurised_data, f, indent=4)

    BuildDataset(
        featuriser.featurised_data,
        fixtures_json_path="preprocessed_data/fixtures.json",
        output_path_scores="preprocessed_data/scores_combined.csv",
        output_path_goals="preprocessed_data/goals_scores.csv"
    )