import requests
import pandas as pd
from functools import reduce

#getting chicago open data
#business xqx5-8hwx
#blocks bt9m-d2mf

'''
os = 0
params = {'$limit': 1000, '$offset': os}
response = requests.get('https://data.cityofchicago.org/resource/xqx5-8hwx.json', params).json()
total_response = response
os = 1000
while len(response) >= 1000:
    params = {'$limit': 1000, '$offset': os}
    response = requests.get('https://data.cityofchicago.org/resource/xqx5-8hwx.json', params).json()
    os += 1000
    total_response += response

business_df = pd.DataFrame.from_dict(total_response)
business_df.columns

to_drop_lst = []
for col in business_df.columns:
  if "computed" in col:
    to_drop_lst.append(col)
business_df.drop(to_drop_lst, axis=1, inplace=True )
'''
#getting ACS data

acs_variables = pd.read_csv('acs_variables.csv')
ACS_API = "https://api.census.gov/data/2017/acs/acs5"

def get_acs_data(df):
    parameters = {
        "get": ",".join(var for var in df.Variable.to_list()),
        "for": "block group:*",
        "in": "state:17 county:031"} # Cook County, Illinois
    request = requests.get(ACS_API, params=parameters)
    data = request.json()
    cols = data.pop(0)[-4:]
    return pd.DataFrame(data, columns=df.Description.to_list() + cols)

def compile_census_data(acs_variables):
    LOCATION_VARIABLES = ["state", "county", "tract", "block group"]
    data = []
    for i in range(acs_variables.Dataset.max()):
        current = acs_variables[acs_variables['Dataset'] == i+1]
        data.append(get_acs_data(current))
    merged = reduce(lambda left, right: pd.merge(left, right, how='outer', on=LOCATION_VARIABLES), data)
    merged["block_group"] = merged.apply(lambda row: "".join(str(row[var]) for var in LOCATION_VARIABLES), axis=1)
    return merged.drop(columns=LOCATION_VARIABLES)
        
census_data = compile_census_data(ACS_VARIABLES)