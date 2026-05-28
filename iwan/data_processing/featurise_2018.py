from typing import Dict, List
import pandas as pd

prem_team_map = {
    "Arsenal": "Arsenal",
    "Brighton & Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Huddersfield Town": "Huddersfield Town",
    "Leicester City": "Leicester City",
    'Liverpool': "Liverpool",
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    'Southampton': "Southampton",
    "Stoke City": "Stoke City",
    'Swansea City': "Swansea City",
    'Tottenham Hotspur': "Tottenham Hotspur",
    "Watford": "Watford",
    'West Bromwich Albion': 'West Bromwich Albion',
    'West Ham United': "West Ham United"
}

europe_maps = {
    "league_df": "squad_df",
    "ligue_1": {
        "Paris SG": "Paris Saint-Germain",
    },
    "serie_a": {
        "Inter": "Inter Milan",
    },
    "bundesliga": {
        "Dortmund": "Borussia Dortmund",
        'Ein Frankfurt': 'Eintracht Frankfurt',
        'FC Koln': '1. FC Köln',
        "Hannover": "Hannover 96",
        "Hertha": "Hertha BSC",
        'Leverkusen': 'Bayer Leverkusen',
        "M'gladbach": 'Borussia Mönchengladbach',
        "Mainz": 'Mainz 05'
    },
    "laliga": {
        'Alaves': 'Alavés',
        'Ath Bilbao': 'Athletic Bilbao',
        'Ath Madrid': 'Atlético Madrid',
        'Betis': 'Real Betis',
        "Celta": 'Celta Vigo',
        "Espanol": 'Espanyol',
        'La Coruna': 'Deportivo La Coruña',
        'Leganes': 'Leganés',
        'Malaga': 'Málaga',
        'Sociedad': 'Real Sociedad'
    }
}

def insert_2018_data(
    featurised_squads: Dict, 
    league_dfs: Dict[str, Dict[str, pd.DataFrame]],
    top_domestic: Dict[str, Dict[str, List[str]]],
    goalscorers: Dict[str, Dict[str, List[str]]],
    csv_path="raw_data/2018 FIFA World Cup Squads.csv",
) -> Dict:

    print(featurised_squads.keys())
    wc_teams = pd.read_csv(csv_path)
    countries = wc_teams["Team"].unique()
    top_scorers = []
    for league, scorers in goalscorers["2018"].items():
        top_scorers.extend(scorers)

    for country in countries:
        c = country.lower()
        featurised_squads[2018][c] = {}
        featurised_squads[2018][c]["is_home_nation"] = 0 if country != "Russia" else 1

        # prem
        featurised_squads[2018][c]["premier_league_players"] = 0
        team_df = wc_teams.loc[wc_teams["Team"] == country]
        featurised_squads[2018][c]["n_unique_clubs"] = len(team_df["Club"].unique())
        featurised_squads[2018][c]["players_finishing_top_three_europe_domestic"] = 0
        featurised_squads[2018][c]["top_goalscorers_count"] = 0
        prem_teams = list(prem_team_map.keys())
        for index, row in team_df.iterrows():
            club = row["Club"].strip()
            if row["Club"].strip() in prem_teams:
                featurised_squads[2018][c]["premier_league_players"] += 1
            featurise_europe_players(club, league_dfs, featurised_squads[2018][c])
            increment_top_three_domestic(top_domestic, club, featurised_squads[2018][c])
            increment_goal_scorers(row["Player"], top_scorers, featurised_squads[2018][c])
    
    return featurised_squads


def featurise_europe_players(club: str, league_dfs: Dict[str, Dict[int, pd.DataFrame]], output_dict: Dict):
    # Change featurise_squads.py to store dfs in dict for simplicity
    # iterate through leagues and count players
    for league in ["laliga", "serie_a", "bundesliga", "ligue_1"]:
        if output_dict.get(f"{league}_players") is None:
            output_dict[f"{league}_players"] = 0
        df = league_dfs[league][2018]
        league_teams = list(df["HomeTeam"].unique())
        league_teams = [europe_maps[league].get(l, l).lower() for l in league_teams]
        if club.strip().lower() in league_teams:
            output_dict[f"{league}_players"] += 1

def increment_top_three_domestic(top_domestic: Dict[str, Dict[str, List[str]]], club: str, output_dict: Dict):
    changes = {
        "Inter Milan": "Internazionale"
    }
    club = changes.get(club, club)
    league_winners = top_domestic["2018"]
    for league, top_clubs in league_winners.items():
        top_clubs = [c.lower().replace("-", " ") for c in top_clubs]

        if club.strip().lower().replace("-", " ") in top_clubs:
            output_dict["players_finishing_top_three_europe_domestic"] += 1
        else:
            for c in top_clubs:
                if c in club.strip().lower().replace("-", " "):
                    output_dict["players_finishing_top_three_europe_domestic"] += 1


def increment_goal_scorers(name: str, top_scorers: List[str], out_dict: Dict):
    name = name.replace("(captain)", "").strip()
    for scorer in top_scorers:
        if scorer == name:
            out_dict["top_goalscorers_count"] += 1
            return