import json
from typing import Dict, List, Optional
import pandas as pd

class BuildDataset:
    MAP = {
        "fixture alias": "featurised_squads name",
        "côte d'ivoire": "cote d ivoire",
        "côte": "cote d ivoire",
        "bosnia-herzegovina": "bosnia herzegovina",
        "usa": "united states",
        "africa": "south africa",
        'and tobago': 'trinidad tobago',
        'and montenegro': "serbia",

    }

    def __init__(
        self, 
        featurised_squads: Dict[str, Dict[str, int]],
        fixtures_json_path: str="preprocessed_data/fixtures.json",
        output_path_scores: Optional[str]="preprocessed_data/scores_combined.csv",
        output_path_goals: Optional[str]="preprocessed_data/goals_scores.csv"
    ):
        with open(fixtures_json_path, "r") as f:
            all_fixtures = json.load(f)

        scores, goals = self.main(featurised_squads, all_fixtures)
        self.score_df = pd.DataFrame(scores)
        if output_path_scores is not None:
            self.score_df.to_csv(output_path_scores, index=False)

        self.goals_df = pd.DataFrame(goals)
        if output_path_goals is not None:
            self.goals_df.to_csv(output_path_goals, index=False)

    def main(self, featurised_squads: Dict[str, Dict[str, int]], all_fixtures: Dict[str, List[Dict]]):
        results = []
        results_target_team = []
        for year in featurised_squads.keys():
            if str(year) in ["2026"]:
                continue
            fixtures = all_fixtures[str(year)]
            squad_features = featurised_squads[year]
            for idx, fixture in enumerate(fixtures):
                results.append(self.process_fixture(fixture, squad_features, year))
                results_target_team.extend(self.process_fixture_goals(fixture, squad_features, year, idx))
        return results, results_target_team

    def process_fixture(self, fixture: Dict[str, str], squad_features: Dict[str, int], year: str):
        t1 = self.get_team_names(fixture["team_1"])
        t2 = self.get_team_names(fixture["team_2"])
        is_knockout = 1 if fixture["phase"] == "knockout" else 0
        result = {"year": int(year), "is_knockout": is_knockout, "team_1_score": fixture["team_1_score"], "team_2_score": fixture["team_2_score"]}
        result.update(self.get_team_features(t1, squad_features, "1"))
        result.update(self.get_team_features(t2, squad_features, "2"))
        return result
    
    def process_fixture_goals(self, fixture, squad_features, year, idx):
        is_knockout = 1 if fixture["phase"] == "knockout" else 0
        results = []
        for team in ["team_1", "team_2"]:
            opposition = "team_2" if team == "team_1" else "team_1"
            target_name = self.get_team_names(fixture[team])
            opp_name = self.get_team_names(fixture[opposition])
            result = {
                "idx": idx,
                "year": int(year), 
                "is_knockout": is_knockout, 
                "target_team": target_name, 
                "opposition": opp_name, 
                "goals_scored": fixture[f"{team}_score"]
            }
            result.update(self.get_team_features(target_name, squad_features, "1", include_name=False))
            result.update(self.get_team_features(opp_name, squad_features, "2", include_name=False))
            results.append(result)
        return results


    def get_team_names(self, name: str):
        name = name.lower()
        return self.MAP.get(name, name)
    
    def get_team_features(self, name, squad_features: Dict[str, int], team_number:str="1", include_name=True):
        result = {f"team_{team_number}_name": name} if include_name else {}
        try:
            features = squad_features[name]
        except KeyError:
            if name == "united states":
                try:
                    features = squad_features["usa"]
                except:
                    raise KeyError(f"Could not find fixture name '{name}' in squad features.\nAvailable countries are {list(squad_features.keys())}")
            else:
                raise KeyError(f"Could not find fixture name '{name}' in squad features.\nAvailable countries are {list(squad_features.keys())}")
        for key, value in features.items():
            result[f"team_{team_number}_{key}"] = value
        return result
    




