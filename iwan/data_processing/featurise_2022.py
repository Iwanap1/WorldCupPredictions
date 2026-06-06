from .featurise_2018 import featurise_europe_players
import pandas as pd
from typing import List, Dict

mapped_teams = {
    "prem_matches team": "wc team",
    "premier_league": {
        "Manchester Utd": "Manchester United",
        "Newcastle Utd": "Newcastle United",
        "Tottenham": "Tottenham Hotspur",
        "Brighton": "Brighton & Hove Albion",
        "Wolves": "Wolverhampton Wanderers",
        "West Ham": "West Ham United"
    },
    "laliga": {
        "Cadiz": "Cádiz",
        "Celta": "Celta Vigo",
        "Betis": "Real Betis",
        "Espanol": "Espanyol",
        "Ath Bilbao": "Athletic Bilbao",
        "Sociedad": "Real Sociedad",
        "Ath Madrid": "Atlético Madrid",
        "Vallecano": "Rayo Vallecano"
    },
    "bundesliga": {
        "Dortmund": "Borussia Dortmund",
        'Ein Frankfurt': 'Eintracht Frankfurt',
        'FC Koln': '1. FC Köln',
        "Hannover": "Hannover 96",
        "Hertha": "Hertha BSC",
        'Leverkusen': 'Bayer Leverkusen',
        "M'gladbach": 'Borussia Mönchengladbach',
        "Mainz": 'Mainz 05',
        "Wolfsburg": 'VfL Wolfsburg',
        "Stuttgart": 'VfB Stuttgart',
        "Hoffenheim": "1899 Hoffenheim",
        "Freiburg": "SC Freiburg",
        "Augsburg": "FC Augsburg"
    },
    "serie_a": {
        "Inter": "Internazionale",
        "Verona": "Hellas Verona"
    },
    "ligue_1": {
        "Paris SG": "Paris Saint-Germain",
    },
}

prem_teams = ['Brentford',
 'Manchester Utd',
 'Leicester City',
 'Burnley',
 'Chelsea',
 'Watford',
 'Everton',
 'Norwich City',
 'Newcastle Utd',
 'Tottenham',
 'Liverpool',
 'Aston Villa',
 'Manchester City',
 'Leeds United',
 'Crystal Palace',
 'Brighton',
 'Wolves',
 'Southampton',
 'Arsenal',
 'West Ham']

def insert_2022_data(
    featurised_squads: Dict, 
    league_dfs: Dict[str, Dict[str, pd.DataFrame]],
    top_domestic: Dict[str, Dict[str, List[str]]],
    goalscorers: Dict[str, Dict[str, List[str]]],
    csv_path="raw_data/archive/2022_FIFA_World_Cup_squads.csv",
) -> Dict:
    squad_df = pd.read_csv(csv_path)
    countries = squad_df["Country"].unique().tolist()
    outdict = {}
    top_scorers = []
    for _, scorers in goalscorers["2022"].items():
        top_scorers.extend(scorers)

    winning_clubs = []
    for _, teams in top_domestic["2022"].items():
        winning_clubs.extend(teams)

    for c in countries: 
        c_df = squad_df[squad_df["Country"] == c]
        c = c.lower()
        is_home_nation = 1 if c == "qatar" else 0
        outdict[c] = {"is_home_nation": is_home_nation, "top_goalscorers_count": 0,"players_finishing_top_three_europe_domestic": 0, "premier_league_players": 0}
        clubs = c_df["Club"].tolist()
        players = [p.replace("*(captain)*", "") for p in c_df["Player"].unique().tolist()]
        for club in clubs:
            if club in winning_clubs:
                outdict[c]["players_finishing_top_three_europe_domestic"] += 1
            featurise_europe_players(club, league_dfs, outdict[c], mapped_teams, 2022)
            for p_team in prem_teams:
                if mapped_teams["premier_league"].get(p_team, p_team) == club:
                    outdict[c]["premier_league_players"] += 1
                    continue
        for p in players:
            if p in top_scorers:
                outdict[c]["top_goalscorers_count"] += 1
        outdict[c]["n_unique_clubs"] = len(set(clubs))
    featurised_squads[2022] = outdict
    return featurised_squads