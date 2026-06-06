from .featurise_2018 import insert_2018_data
from .featurise_2022 import insert_2022_data
from typing import Dict, List
import pandas as pd
import os, json, re


class FeaturiseSquads:
    MIN_YEAR = 1966
    OVERRIDES = {
        "la_liga": {
            "csv_entry": "parsed_squad_entry",
            "Almeria": "Almería",
            "Malaga": "Málaga",
            "Santander": "Racing Santander",
            "Ath Madrid": "Atlético Madrid",
            "Ath Bilbao": "Athletic Bilbao",
            "Sp Gijon": "Sporting Gijón",
            "La Coruna": 'Deportivo La Coruña',
            "Espanol": "Espanyol",
        },
        "serie_a": {
            "Inter": "Internazionale",
        },
        "bundesliga": {
            "M'gladbach": "Borussia Mönchengladbach",
            'Ein Frankfurt': "Eintracht Frankfurt",
            'Nurnberg': "Nuremberg",
            "FC Koln": "Köln"
        },
        "ligue_1": {
            'Paris SG': "Paris Saint Germain",
            "St Etienne": "Saint Etienne"
        }
    }

    def __init__(
            self, 
            parsed_squads: Dict, 
            prem_matches_csv_path: str = "raw_data/prem_matches.csv",
            eighteen_squads_csv_path: str = "raw_data/2018 FIFA World Cup Squads.csv",
            twenty_two_squads_csv_path: str = "raw_data/archive/2022_FIFA_World_Cup_squads.csv",
            laliga_dir: str = "raw_data/la_liga/",
            serie_a_dir: str = "raw_data/serie_a/",
            bundesliga_dir: str = "raw_data/bundesliga/",
            ligue_dir: str = "raw_data/ligue_1/", 
            min_dir_path="../data/worldcup/min",
            world_rankings_json_path="raw_data/world_rankings.json",
            league_winners_json_path="raw_data/domestic_league_winners.json",
            goal_scorer_json_path="raw_data/domestic_top_scorers.json"
        ):

        self.parsed_squads = parsed_squads

        # load data
        self.prem_matches_df = pd.read_csv(prem_matches_csv_path)
        self.league_dfs = self.make_df_league_dict(laliga_dir, serie_a_dir, bundesliga_dir, ligue_dir)
        with open(league_winners_json_path, "r") as f:
            self.league_winners = json.load(f)
        with open(goal_scorer_json_path, "r") as f:
            self.goalscorers_json = json.load(f)
        with open(world_rankings_json_path, "r") as f:
            self.rankings = json.load(f)

        # main
        self.featurise_early()

        self.featurised_data = insert_2018_data(
            self.featurised_data, 
            self.league_dfs,
            self.league_winners,
            self.goalscorers_json,
            csv_path=eighteen_squads_csv_path,
        )
        self.featurised_data = insert_2022_data(
            self.featurised_data, 
            self.league_dfs,
            self.league_winners,
            self.goalscorers_json,
            csv_path=twenty_two_squads_csv_path,
        )
        self.extract_made_knockouts(min_dir_path)
        self.add_world_rankings()


    def featurise_early(self):
        self.featurised_data = {}
        for year in self.parsed_squads.keys():
            if int(year) < self.MIN_YEAR:
                continue
            self.featurised_data[year] = {}
            for team in self.parsed_squads[year]["teams"].keys():
                team_data = {"is_home_nation": self.get_home_nation(year, team)}
                players = self.parsed_squads[year]["teams"][team]["players"]
                team_data["top_goalscorers_count"] = self.check_goalscorers(year, players)
                team_data["premier_league_players"] = self.count_prem_players(year, players)       
                team_data["laliga_players"] = self.count_from_df_dict(self.league_dfs["laliga"], self.OVERRIDES["la_liga"], year, players) 
                team_data["serie_a_players"] = self.count_from_df_dict(self.league_dfs["serie_a"], self.OVERRIDES["serie_a"], year, players)
                team_data["bundesliga_players"] = self.count_from_df_dict(self.league_dfs["bundesliga"], self.OVERRIDES["bundesliga"], year, players)
                team_data["ligue_1_players"] = self.count_from_df_dict(self.league_dfs["ligue_1"], self.OVERRIDES["ligue_1"], year, players)
                team_data["players_finishing_top_three_europe_domestic"] = self.count_top_three_league_finishers(year, players)
                team_data["n_unique_clubs"] = len(set([player["club"] for player in players]))
                team = self.norm_country_name(team)
                self.featurised_data[year][team] = team_data

    def norm_country_name(self, name: str) -> str:
        name = name.lower()
        if name == "espana":
            return "spain"
        if name == "deutschland":
            return "germany"
        return name
    
    def count_top_three_league_finishers(self, year: int, players: List[Dict]):
        year_data = self.league_winners[str(year)]
        count = 0
        for player in players:
            for league, top_three in year_data.items():
                for team in top_three:
                    if team.lower() in player["club"].lower().replace("-", " "):
                        count += 1
                        break
        return count



    def extract_made_knockouts(self, min_dir_path):
        for feature_year in self.featurised_data.keys():
            query_year = int(feature_year) - 4
            with open(f"{min_dir_path}/{query_year}.txt", "r") as f:
                text = f.read()
            for country in self.featurised_data[feature_year].keys():
                for feature_name in ["qualified_last_time", "made_r16_last_time", "made_quarters_last_time", "made_semis_last_time", "made_final_last_time"]:
                    feature_value = self.made_knockout_round_last_time(text, country, phase=feature_name)
                    self.featurised_data[feature_year][country][feature_name] = feature_value
    
    def count_prem_players(self, year, players: List[Dict[str, str]]) -> int:
        manual_overrides = {
            'qpr': "queens park rangers",
            'sheffield weds': "sheffield wednesday",
            "west brom": "west bromwich albion",
            "nott'ham forest": "nottingham forest",
        }
        year = int(year)
        previous_year = year - 1
        year_key = f"{previous_year}/{year}"
        filtered_matches = self.prem_matches_df[self.prem_matches_df['Season'] == year_key]
        teams = filtered_matches["Home"].str.lower().unique().tolist()
        count = 0
        for player in players:
            for team in teams:
                if str(team) in player["club"].lower() or str(team).replace("utd", "united") in player["club"].lower() or str(team).replace("ath", "athletic") in player["club"].lower():
                    count += 1
                else:
                    override = manual_overrides.get(team)
                    if override and override in player["club"].lower():
                        count += 1
                        break
        return count
    

    def count_from_df_dict(self, df_dict: Dict[int, pd.DataFrame], overrides, year: int, players: List[Dict]) -> int:
        teams = df_dict[year]["HomeTeam"].unique().tolist()
        count = 0
        for player in players:
            for team in teams:
                team_name = overrides.get(team, team)
                if team_name.lower() in player["club"].lower().replace("-", " "):
                    count += 1
        
        return count
    
    def get_home_nation(self, year, team):
        if int(year) == 2002:
            if team in ["south korea", "japan"]: 
                return 1
            else:
                return 0
        
        if int(year) == 1994:
             if team in ["usa", "united states"]:
                return 1
             else:
                return 0
            
        if team == self.parsed_squads[year]["home_nation"]:
            return 1
        
        return 0
    
    def make_df_league_dict(
        self, 
        laliga_dir: str, 
        serie_a_dir: str, 
        bundesliga_dir: str, 
        ligue_dir: str
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        return {
            "laliga": self.load_csvs_from_dir(laliga_dir),
            "serie_a": self.load_csvs_from_dir(serie_a_dir),
            "bundesliga": self.load_csvs_from_dir(bundesliga_dir),
            "ligue_1": self.load_csvs_from_dir(ligue_dir)
        }


    def load_csvs_from_dir(self, dir_path: str) -> Dict[int, pd.DataFrame]:
        dfs = {}
        for file in os.listdir(dir_path):
            if file.endswith(".csv"):
                try:
                    df_name = int(file.split(".")[0])
                except:
                    raise Exception(f"Unexpected file name in {dir_path}: {file}. All files should be named as <year>.csv")
                dfs[df_name] = pd.read_csv(f"{dir_path}/{file}")
        return dfs
    
    def made_knockout_round_last_time(self, text, country, phase="made_final_last_time"):
        phase_map = {
            "made_r16_last_time": ["Round of 16", "Quarter-final"],
            "made_quarters_last_time": ["Quarter-final", "Semi-finals"],
            "made_semis_last_time": ["Semi-final", "Final"]
        }
        if phase == "qualified_last_time":
            if country.lower() in text.lower():
                return 1
            if country.lower().replace("-", "") in "united states" and "usa" in text.lower():
                return 1
            return 0

        if phase in phase_map:
            processed = text.split(phase_map[phase][0])[1].split(phase_map[phase][1])[0].strip().lower()
            if country.lower() in processed:
                return 1
            if country.lower().replace("-", "") in "united states" and "usa" in processed:
                return 1
            return 0
        elif phase == "made_final_last_time":
            processed = text.split("Final")[1].strip().lower()
            if country.lower() in processed:
                return 1
            if country.lower().replace("-", "") in "united states" and "usa" in processed:
                return 1
            return 0
        else:
            raise ValueError("Invalid phase specified")
        
    def check_goalscorers(self, year: int, players: List[Dict]):
        league_scorers = self.goalscorers_json[str(year)]
        scorers_list = []
        for key, scorer_list in league_scorers.items():
            scorers_list.extend(scorer_list)

        player_name_list = [p["name"] for p in players]
        count = 0
        for scorer in scorers_list:
            if scorer == "Ronaldo": 
                if "Ronaldo" in player_name_list:
                    count += 1
                    continue
            else:
                for p in player_name_list:
                    if scorer in p:
                        count += 1
                        break
        return count
                
    def get_name_and_rank_dict(self, year):
        # South Korea (37)
        mapped_names = {
            "fr yugoslavia": "yugoslavia",
            "republic of ireland": "ireland",
            "bosnia and herzegovina": "bosnia herzegovina",
            "ivory coast": "cote d ivoire",
            "trinidad and tobago": "trinidad tobago",
            "serbia and montenegro": "serbia"
        }
        res = {}
        for name_rank in self.rankings[str(year)]:
            number = re.findall(r"\d+", name_rank)[0]
            name = re.sub(r"\(\d+\)", "", name_rank).strip().lower()
            name = mapped_names.get(name, name)
            res[name] = int(number)
        return res

    def add_world_rankings(self):
        for year in self.featurised_data.keys():
            if str(year) in ["2026"]:
                continue
            name_rank_dict = self.get_name_and_rank_dict(year)
            for country, data in self.featurised_data[year].items():
                self.featurised_data[year][country]["ranking"] = name_rank_dict[country]