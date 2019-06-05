import requests
import pandas as pd
from functools import reduce
import os.path

def get_chicago_data():
    '''
    API requests the City of Chicago data portal to create 3 CSVs of the following information:
        - business licenses
        - block groups
        - 311 call log
    '''
    CSV_URL = 'https://data.cityofchicago.org/api/views/'
    CSV_FRAG = '/rows.csv?accessType=DOWNLOAD'
    URLS = {'business': CSV_URL + 'xqx5-8hwx' + CSV_FRAG,
            'blocks': CSV_URL + 'bt9m-d2mf' + CSV_FRAG,
            '311': CSV_URL + 'v6vf-nfxy' + CSV_FRAG,
            'crimes': CSV_URL + 'ijzp-q8t2' + CSV_FRAG} #datasets to download
    for key, val in URLS.items():
        if not os.path.isfile(key + '.csv'): #check if already downloaded
            df = pd.read_csv(val)
            print(key, 'download complete')
            if key == 'business':
                df = df[df['CITY'] == 'CHICAGO']
            if key == 'crimes':
                df = df[df['Date'] > '1/1/2010']
            df.to_csv(key + '.csv')
            print(key, 'saved')
        else:
            print(key, 'already downloaded')


#getting ACS data
def get_acs_data(df, ACS_API):
    '''
    '''
    parameters = {
        "get": ",".join(var for var in df.Variable.to_list()),
        "for": "block group:*",
        "in": "state:17 county:031"} # Cook County, Illinois
    try:
        request = requests.get(ACS_API, params=parameters)
        data = request.json()
        cols = data.pop(0)[-4:]
        return pd.DataFrame(data, columns=df.Description.to_list() + cols)
    except:
        return None


def compile_census_data(acs_variables, year):
    '''
    '''
    LOCATION_VARIABLES = ["state", "county", "tract", "block group"]
    ACS_API = "https://api.census.gov/data/" + str(year) + "/acs/acs5"
    data = []
    for i in range(acs_variables['Dataset'].max()): #loop over ACS tables
        current = acs_variables[acs_variables['Dataset'] == i+1]
        print(current)
        result = get_acs_data(current, ACS_API)
        if result:
            data.append(result)
    merged = reduce(lambda left, right: pd.merge(left, right, how='outer', on=LOCATION_VARIABLES), data)
    merged["block_group"] = merged.apply(lambda row: "".join(str(row[var]) for var in LOCATION_VARIABLES), axis=1)
    return merged.drop(columns=LOCATION_VARIABLES)


def save_census_data():
    '''
    '''
    acs_variables = pd.read_csv('acs_variables.csv')
    acs_17 = compile_census_data(acs_variables, 2017)
    acs_13 = compile_census_data(acs_variables, 2013)
    acs_17.to_csv('acs_17.csv')
    print('saved 17')
    acs_13.to_csv('acs_13.csv')
    print('saved 13')
