import process_data as pr
import pandas as pd
import numpy as numpy

def impute_lat_long(bus_df):
    '''
    '''
    missing_data = bus_df[(bus_df['CITY'] == 'CHICAGO') & (bus_df['LATITUDE'].isnull())]
    missing_dict = {}
    account_lst = []
    for account in missing_data['ACCOUNT NUMBER'].unique():
        missing_dict[account] = []
        account_lst.append(account)
    for key in missing_dict.keys():
        bus_df['ACCOUNT NUMBER'] = bus_df['ACCOUNT NUMBER'].astype(str)
        i = 0
        if str(bus_df[bus_df['ACCOUNT NUMBER'] == key].iloc[i]['LATITUDE']) != 'nan':
            missing_dict[key] = bus_df[bus_df['ACCOUNT NUMBER'] == key].iloc[i][['LATITUDE', 'LONGITUDE']]
        i += 1
    return missing_dict