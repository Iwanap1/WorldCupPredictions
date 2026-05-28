from typing import Dict
import pandas as pd

class AngusModel:
    def __init__(self):
        pass

    def predict(fixtures_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
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


        
