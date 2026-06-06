import re
import os
import json
import pandas as pd



country_dict = {
    'korea': 'korea republic',
    'africa': 'south africa',
    'united states': 'usa',
}


def extract_cup_data(file_path: str):
    """ takes cup.txt file and creates json file with the data
    format as follows: 
        {
        tournament_name: str,
        year: int,
        teams: list[str],
        groups: list[dict],
        matches: list[dict]
        }  """

    with open(file_path, 'r') as file:
        data = file.read()
    
   
    # parse the data
    line_1 = data.split('\n')[0].strip()
    tourn_regex = r'=(.*) # (.*)'
    tournament_name = re.match(tourn_regex, line_1).group(1)
    year = int(tournament_name.split(' ')[3])
    
   

    # parse groups
    group_data = []
    group_regex = r'Group (.*) + \| (.*)'
    groups = re.finditer(group_regex, data)
    for group in groups:
        teams = group.group(2).split('    ')
        group_data.append({'group_name': group.group(1), 'teams': [team.strip() for team in teams]})
    
   
    # parse group results
    group_results = []


    results_blocks = data.split('▪ ')

    results = []

    for results_block in results_blocks:
       
        result_block_name = ''
        title_regex = r'Group +.*|Quarter-finals|Semi-finals|Third-place match|Final'

        title_match = re.search(title_regex, results_block)
      
        if title_match:
            result_block_name = title_match.group(0).strip()
        else:
            continue
        for line in results_block.split('\n'):
            if line.strip() == '':
                continue
        
            
            
            # remove day of week from line if it exists
            line = re.sub(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b', '', line)
            # remove (int-int) from line if it exists
            line = re.sub(r'\(\d+-\d+\)\s', '', line)

            # remove time if exists i.e 18:00 
            line = re.sub(r'\d{2}:\d{2}', '', line)
    
                
            
            # line of form dd Month  Country_1  int-int  Country_2  @ stadium, city
            # or of form Month dd  Country_1  int-int  Country_2  @ stadium, city

            line_regex = r'(\d{1,2} [A-Za-z]+ | [A-Za-z]+ \d{1,2} ) (.*) (\d+-\d+) (.*) @ (.*), (.*)'
            line_match = re.match(line_regex, line)
            if not line_match:
                if '@' in line:
                    try:
                        if result_block_name:
                            # remove utc from line if it exists
                            line = re.sub(r'UTC\+\d{1,2}', '', line)
                            # remove to the right of @
                            line = line.split('@')[0]
                            
                            # take score as \d+-\d+
                            score = re.search(r'\d+-\d+', line).group(0)
                            left_of_score = line.split(score)[0]
                            left_of_score_reversed = ' '.join(left_of_score.split(' ')[::-1])
                            right_of_score = line.split(score)[1]
                            right_of_score_reversed = ' '.join(right_of_score.split(' ')[::-1])
                            # team 1 is first occuring country in line
                            team_1 = re.search(r'[A-Za-z]+', left_of_score_reversed).group(0)
                            # team 2 is second occuring country in line
                            team_2 = re.search(r'[A-Za-z]+', right_of_score_reversed).group(0)
                            # stadium is last occuring stadium in line
                            stadium = None
                            # city is last occuring city in line
                            city = None
                            date = None
                            results.append({'result_block_name': result_block_name, 'date': date, 'team_1': team_1, 'score': score, 'team_2': team_2, 'stadium': stadium, 'city': city})
                            continue
                    except:
                        continue
                print(line)
                continue

            print(line_match.groups())

            d1 = line_match.group(1)
            d2 = line_match.group(2)
            if d1:
                date = d1.strip()
            elif d2:
                date = d2.split(' ')[-1] + ' ' + d2.split(' ')[0]
                date = date.strip()
            
            

            team_1 = line_match.group(2).strip()

            score = line_match.group(3).strip()
            team_2 = line_match.group(4).strip()
            stadium = line_match.group(5).strip()
            city = line_match.group(6).strip()
        
            results.append({'result_block_name': result_block_name, 'date': date, 'team_1': team_1, 'score': score, 'team_2': team_2, 'stadium': stadium, 'city': city})
        
    return results
    

def score_parser(score_string: str):
    """ takes a score string and returns a tuple of the score """
    score_regex = r'(\d+)-(\d+)'
    score = re.match(score_regex, score_string).groups()
    return int(score[0]), int(score[1])

def loop_through_worldcup_files():
    """ loops through the worldcup files and extracts the data """
    results = {}
    for worldcup_folder in os.listdir('/users/angus/cursor/WorldCupPredictions/data/worldcup'):
        if re.match(r'\d{4}--.*', worldcup_folder):
            print(f'Extracting data from {worldcup_folder}')
            cup_data = extract_cup_data(f'/users/angus/cursor/WorldCupPredictions/data/worldcup/{worldcup_folder}/cup.txt')
            title = worldcup_folder
            location = worldcup_folder.split('--')[1]
            year = int(worldcup_folder.split('--')[0])
    
            results[worldcup_folder] = {'year': year, 'location': location, 'data': cup_data}
    return results

def save_results(results: dict):
    """ saves the results to a json file """
    with open('results.json', 'w') as file:
        json.dump(results, file)

def get_rankings_df():
    """ gets the rankings dataframe """
    rankings_df = pd.read_csv('/users/angus/cursor/WorldCupPredictions/angus/additional_data/fifa_mens_rank.csv')
    return rankings_df

def get_results_dict():
    """ gets the results dictionary """
    with open('results.json', 'r') as file:
        results = json.load(file)
    return results

def create_ranking_results_dataset(results: dict, rankings_df: pd.DataFrame):
    ranking_results = []
    for tourn in results.values():
        year = tourn['year']
        if year < 1990:
            continue
        print('year', year)
        print('results', tourn['data'])
      
        for match in tourn['data']:
           
            team_1 = match['team_1']
            team_2 = match['team_2']
            score_1 = match['score'].split('-')[0]
            score_2 = match['score'].split('-')[1]
       
            country_rankings = rankings_df[rankings_df['team'].str.lower() == team_1.lower()]
           
            ranking = country_rankings[country_rankings['date'] == year]['rank'].values
           

            try:
                print(year)
                country_rankings = rankings_df[rankings_df['team'].str.lower() == team_1.lower()]              
                ranking_1 = country_rankings[country_rankings['date'] == year]['rank'].values[0]
        
            except:
                ranking_1 = None
            try:
                country_rankings = rankings_df[rankings_df['team'].str.lower() == team_2.lower()]
                ranking_2 = country_rankings[country_rankings['date'] == year]['rank'].values[0]
            except:
                ranking_2 = None
            
            ranking_results.append({'year': year, 'team_1': team_1, 'team_2': team_2, 'score_1': score_1, 'score_2': score_2, 'ranking_1': ranking_1, 'ranking_2': ranking_2})
     
 
    return ranking_results



if __name__ == '__main__':

  

    files = loop_through_worldcup_files()
    save_results(files)

    rankings_df = get_rankings_df()
    results = get_results_dict()
    clean_results = clean_results(results)
    ranking_results = create_ranking_results_dataset(results, rankings_df)
    df = pd.DataFrame(ranking_results)
    print(df.head())

    df.to_csv('ranking_results.csv', index=False)