import os, re


class SquadParser:
    POSITION_MAP = {
        "GK": "Goalkeeper",
        "DF": "Defender",
        "MF": "Midfielder",
        "FW": "Forward"
    }

    def __init__(self, data_dir="../data/worldcup", min_year=1966):
        self.data_dir = data_dir
        self.min_year = min_year
        self.parse_squads()

    def parse_squads(self):
        self.parsed_squads = {}
        for wc in os.listdir(self.data_dir):
            try:
                year, home_nation = wc.split("--")
            except:
                continue

            year = int(year)
            if year < self.min_year:
                continue
            self.parsed_squads[year] = {"home_nation": home_nation}
            self.parsed_squads[year]["teams"] = {}
            try:
                squads = os.listdir(f"{self.data_dir}/{wc}/squads")
            except:
                continue

            for code_country in squads:
                try:
                    code, country = self.get_code_and_country(code_country)
                except:
                    continue

                self.parsed_squads[year]["teams"][country] = {"players": []}
                with open(f"{self.data_dir}/{wc}/squads/{code_country}", "r") as f:
                    for line in f.readlines():
                        self.parse_line(line, year, country)
                        

    def parse_line(self, line, year, country):
        line = line.strip()
        if not re.match(r"\(\d+\)", line):
            return
        # example line:  "(1)  GK  Mathew Ryan                        ##   6, Club Brugge (BEL)""
        player = {}
        player["number"] = int(re.findall(r"\((\d+)\)", line)[0])
        positions = list(self.POSITION_MAP.keys())

        for pos in self.POSITION_MAP.keys():
            if pos + " " in line:

                if player.get("position") is None:
                    player["position"] = self.POSITION_MAP[pos]
                else:
                    raise Exception(f"Multiple positions found for player: {line}")
        # name = re.findall(r"\d+\)\s+([A-Za-z\s]+)\s+##", line)
        name = re.findall(r"\d+\)\s+(.*)\s+##", line)
        if not name:
            return
        name = name[0]
        for pos in positions:
            name = name.replace(f"{pos}", "")
        player["name"] = name.strip()
        club = line.split(",")[1].strip()
        player["club"] = club

        self.parsed_squads[year]["teams"][country]["players"].append(player)


    def get_code_and_country(self, code_country): 
        split = code_country.replace(".txt", "").split("-")
        if len(split) == 1:
            code = split[0]
            country = split[1]
        else:
            code = split[0]
            country = " ".join(split[1:])
        return code, country