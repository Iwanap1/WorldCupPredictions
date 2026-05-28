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
        output_path: Optional[str]="preprocessed_data/dataset.csv"
    ):
        with open(fixtures_json_path, "r") as f:
            all_fixtures = json.load(f)

        results = self.main(featurised_squads, all_fixtures)
        self.df = pd.DataFrame(results)
        if output_path is not None:
            self.df.to_csv(output_path, index=False)


    def main(self, featurised_squads: Dict[str, Dict[str, int]], all_fixtures: Dict[str, List[Dict]]):
        results = []
        for year in featurised_squads.keys():
            if str(year) in ["2026", "2022"]:
                continue
            fixtures = all_fixtures[str(year)]
            squad_features = featurised_squads[year]
            for fixture in fixtures:
                results.append(self.process_fixture(fixture, squad_features, year))
        return results


    def process_fixture(self, fixture: Dict[str, str], squad_features: Dict[str, int], year: str):
        t1 = self.get_team_names(fixture["team_1"])
        t2 = self.get_team_names(fixture["team_2"])
        is_knockout = 1 if fixture["phase"] == "knockout" else 0
        result = {"year": int(year), "is_knockout": is_knockout}
        result.update(self.get_team_features(t1, squad_features, "1"))
        result.update(self.get_team_features(t2, squad_features, "2"))
        return result

    def get_team_names(self, name: str):
        name = name.lower()
        return self.MAP.get(name, name)
    
    def get_team_features(self, name, squad_features: Dict[str, int], team_number:str="1"):
        result = {f"team_{team_number}_name": name}
        try:
            features = squad_features[name]
        except KeyError:
            print(name)
            raise KeyError(f"Could not find fixture name '{name}' in squad features.\nAvailable countries are {list(squad_features.keys())}")
        for key, value in features.items():
            result[f"team_{team_number}_{key}"] = value
        return result



