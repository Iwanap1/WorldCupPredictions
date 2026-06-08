import pandas as pd
from typing import Dict

class ShafeeModel:

    TEAM_MAP = {
        "IR Iran": "Iran",
        "Curaçao": "Curacao",
        "Cabo Verde": "Cape Verde"
    }
    # mapping of names in 2026 fixture dataset to names in your dataset
    # name = self.TEAM_MAP.get(team_name_in_fixture_df, team_name_in_fixture_df)

    def __init__(self):
        pass
        

    def predict(self, fixtures_to_predict: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """"Must not have any required arguments besides fixtures_df (put those in the __init__), 
        
        Must return a dict like below:
        {
            "<match_id>": {
                "team_1_score": 0,
                "team_2_score": 1
            },
            ...
        }

        fixtures_df has columns: 
            id, match_number, home_team_id, away_team_id, city_id, stage_id, kickoff_at, match_label, team_1_name, team_2_name

            
        use fixtures_df["id"] as the keys for the results.
        """

        # convert the team names in fixtures_to_predict DataFrame to the team names in dataset.csv
        fixtures_to_predict["team_1_name"].apply(lambda name: self.TEAM_MAP.get(name, name))
        fixtures_to_predict["team_2_name"].apply(lambda name: self.TEAM_MAP.get(name, name))

        return

