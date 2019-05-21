import geopandas as gpd
import numpy as np
from shapely.geometry import shape, Point
import shapely.wkt
import pandas as pd
from dateutil.relativedelta import relativedelta


def process_census(acs_csv):
    '''
    Reads in the ACS census data from csv, creating bins for specified features
    and finds percentages for these features

    Inputs: 
        acs_csv: a csv of American Community Survey data

    Output: a pandas dataframe of desired ACS columns
    '''
    df = pd.read_csv(acs_csv)
    total_male = 'Estimate!!Total!!Male'
    total_female = 'Estimate!!Total!!Female'
    total_travel_time = 'Estimate!!Total_x'
    total_inc = 'hhinc_respondents'
    agg_dict = {'male_children': df['Estimate!!Total!!Male!!Under 5 years'] +
                    df['Estimate!!Total!!Male!!5 to 9 years'] +
                    df['Estimate!!Total!!Male!!10 to 14 years'] +
                    df['Estimate!!Total!!Male!!15 to 17 years'],
                'male_working': df['Estimate!!Total!!Male!!18 and 19 years'] + 
                    df['Estimate!!Total!!Male!!20 years'] +
                    df['Estimate!!Total!!Male!!21 years'] +
                    df['Estimate!!Total!!Male!!22 to 24 years'] +
                    df['Estimate!!Total!!Male!!25 to 29 years'] +
                    df['Estimate!!Total!!Male!!30 to 34 years'] +
                    df['Estimate!!Total!!Male!!35 to 39 years'] +
                    df['Estimate!!Total!!Male!!40 to 44 years'] +
                    df['Estimate!!Total!!Male!!45 to 49 years'] +
                    df['Estimate!!Total!!Male!!50 to 54 years'] +
                    df['Estimate!!Total!!Male!!55 to 59 years'] +
                    df['Estimate!!Total!!Male!!60 and 61 years'] +
                    df['Estimate!!Total!!Male!!62 to 64 years'],
                'male_elderly':df['Estimate!!Total!!Male!!65 and 66 years'] +
                    df['Estimate!!Total!!Male!!67 to 69 years'] +
                    df['Estimate!!Total!!Male!!70 to 74 years'] +
                    df['Estimate!!Total!!Male!!75 to 79 years'] +
                    df['Estimate!!Total!!Male!!80 to 84 years'] +
                    df['Estimate!!Total!!Male!!85 years and over'],
                'female_chilren': df['Estimate!!Total!!Female!!Under 5 years'] +
                      df['Estimate!!Total!!Female!!5 to 9 years'] + 
                      df['Estimate!!Total!!Female!!10 to 14 years'] +
                      df['Estimate!!Total!!Female!!15 to 17 years'],
                'female_working': df['Estimate!!Total!!Female!!18 and 19 years'] +
                      df['Estimate!!Total!!Female!!20 years'] +
                      df['Estimate!!Total!!Female!!21 years'] +
                      df['Estimate!!Total!!Female!!22 to 24 years'] +
                      df['Estimate!!Total!!Female!!25 to 29 years'] +
                      df['Estimate!!Total!!Female!!30 to 34 years'] +
                      df['Estimate!!Total!!Female!!35 to 39 years'] +
                      df['Estimate!!Total!!Female!!40 to 44 years'] +
                      df['Estimate!!Total!!Female!!45 to 49 years'] +
                      df['Estimate!!Total!!Female!!50 to 54 years'] +
                      df['Estimate!!Total!!Female!!55 to 59 years'] +
                      df['Estimate!!Total!!Female!!60 and 61 years'] +
                      df['Estimate!!Total!!Female!!62 to 64 years'],
                'female_elderly': df['Estimate!!Total!!Female!!65 and 66 years'] +
                      df['Estimate!!Total!!Female!!67 to 69 years'] +
                      df['Estimate!!Total!!Female!!70 to 74 years'] +
                      df['Estimate!!Total!!Female!!75 to 79 years'] +
                      df['Estimate!!Total!!Female!!80 to 84 years'] +
                      df['Estimate!!Total!!Female!!85 years and over'],
                'low_travel_time': df['Estimate!!Total!!Less than 5 minutes'] +
                       df['Estimate!!Total!!5 to 9 minutes'] +
                       df['Estimate!!Total!!10 to 14 minutes'] +
                       df['Estimate!!Total!!15 to 19 minutes'] +
                       df['Estimate!!Total!!20 to 24 minutes'],
                'med_travel_time': df['Estimate!!Total!!25 to 29 minutes'] +
                       df['Estimate!!Total!!30 to 34 minutes'] +
                       df['Estimate!!Total!!35 to 39 minutes'] +
                       df['Estimate!!Total!!40 to 44 minutes'] +
                       df['Estimate!!Total!!45 to 59 minutes'],
                'high_travel_time': df['Estimate!!Total!!60 to 89 minutes'] +
                        df['Estimate!!Total!!90 or more minutes'],
                'below_pov': df['hhinc_00_10K'] + df['hhinc_10_15K'] +
                             df['hhinc_15_20K'] + df['hhinc_20_25K'],
                'below_med': df['hhinc_25_30K'] + df['hhinc_30_35K'] +
                             df['hhinc_35_40K'] + df['hhinc_40_45K'] +
                             df['hhinc_45_50K'] + df['hhinc_50_60K'],
                'above_med': df['hhinc_60_75K'] + df['hhinc_75_100K'],
                'high_inc': df['hhinc_100_125K'] + df['hhinc_125_150K'] + 
                            df['hhinc_150_200K']}
    #we need to add total respondents for this section
    #total_bachelors = 'Estimate!!Total_y'
    desired_cols = ['block_group']
    for col in agg_dict.keys():
        df[col] = agg_dict[col]
        if col in ['male_children', 'male_working', 'male_children']:
            denom = total_male
        if col in ['female_children', 'female_working', 'female_children']:
            denom = total_female
        if col in ['low_travel_time', 'med_travel_time', 'high_travel_time']:
            denom = total_travel_time
        if col in ['below_pov', 'below_med', 'above_med', 'high_inc']:
            denom = total_inc
        new_col = 'pct_' + col
        df[new_col] = df[col] / df[denom] * 100
        desired_cols.append(new_col)
    for col in ['race_white', 'race_black', 'race_asian']:
        new_col = 'pct_' + col
        df[new_col] = df[col] / df['race_respondents'] * 100
        desired_cols.append(new_col)
    return df[desired_cols]



col_types = {'ACCOUNT NUMBER': str, 'SITE NUMBER': int, 'LICENSE CODE': str,
             'ADDRESS': str, 'APPLICATION TYPE': str, 
             'APPLICATION REQUIREMENTS COMPLETE': str,
               'LICENSE TERM START DATE': str , 'LICENSE STATUS' : str,
               'LICENSE TERM EXPIRATION DATE': str, 'DATE ISSUED': str,
               'LONGITUDE': str, 'LATITUDE': str}

def process_business():
    '''
    Reads in business license csv and specifies necessary columns and their
    types

    Inputs: None

    Outputs: a pandas dataframe
    '''
    df = pd.read_csv('business.csv', usecols=list(col_types.keys()), dtype=col_types,
                     parse_dates=['APPLICATION REQUIREMENTS COMPLETE',
                     'LICENSE TERM START DATE', 'LICENSE STATUS',
                     'LICENSE TERM EXPIRATION DATE', 'DATE ISSUED'], infer_datetime_format=True)
    df['unique_id'] = df['ACCOUNT NUMBER'].astype('str') + '-' + df['SITE NUMBER'].astype('str')
    #find_duration(df, 'DATE ISSUED', 'APPLICATION REQUIREMENTS COMPLETE')
    return df


def tmp(bus_df):
    to_write = {new: [] for new in ['unique_id', 'latitude', 'longitude', 'earliest_date_issued', 'latest_date_issued', 'latest_exp_date']}  
    top_codes = bus_df.groupby('LICENSE CODE').size().sort_values(ascending=False)[:20] 
    for code in list(top_codes.index):
        to_write[code] = []
    for bus_id in bus_df['unique_id']:
        to_write['unique_id'].append(bus_id)
        relevant = bus_df[bus_df['unique_id'] == bus_id]
        to_write['earliest_date_issued'].append(relevant['LICENSE TERM START DATE'].min())
        to_write['latest_date_issued'].append(relevant['LICENSE TERM START DATE'].max())
        to_write['latest_exp_date'].append(relevant['LICENSE TERM EXPIRATION DATE'].max())
        to_write['latitude'].append(relevant['LATITUDE'].iloc[-1])
        to_write['longitude'].append(relevant['LONGITUDE'].iloc[-1])
        for code in list(top_codes.index):
            if code in list(relevant['LICENSE CODE']):
                to_write[code].append(1)
            else:
                to_write[code].append(0)
    new_bus_df = pd.DataFrame.from_dict(to_write)
    return new_bus_df


    ##smalldf
    counts = inputdf.groupby('unique_id').size()
    for i, df in minidf.groupby('unique_id'):
        #if address doesn't change
        #store entry and permits
        
        #if address changes
        #? and permits
        pass        
        '''
        cur = {}
        if df['ADDRESS'].unique().size == 1:
            cur['ADDRESS'] = 
        to_write[i] = 
        '''


def find_duration(df, col1, col2):
    '''
    Calculates the amount of days that have passed between two columns

    Inputs: 
        df: a pandas dataframe
        col1: column name of dataframe
        col2: column name of dataframe

    '''
    diff = df[col2] - df[col1]
    df['days_between_' + col1 + "_" + col2] = diff.dt.days


def check_alive(df, date):
    '''
    Checks to see if business still open after given date and gives the
    following encoding for a new column in the dataframe ('status'):
        0: closed
        1: open
    
    Inputs:
        df: a pandas dataframe
        date: a datetime object

    Outputs: None
    '''
    df["status"] = np.nan
    for account in df['ACCOUNT NUMBER'].unique():
        if np.where(df[df['ACCOUNT NUMBER'] == account]['LICENSE TERM EXPIRATION DATE'].max() < date, 0, 1) == 1:
            df.loc[df['ACCOUNT NUMBER'] == account, 'status'] = 1
        else:
            df.loc[df['ACCOUNT NUMBER'] == account, 'status'] = 0


def survive_two_years(df):
    '''
    '''
    df['exists_2_yrs'] = np.nan
    for account in df['ACCOUNT NUMBER'].unique():
        if np.where(df[df['ACCOUNT NUMBER'] == account]['LICENSE TERM EXPIRATION DATE'].max() < (df['DATE ISSUED'] + np.timedelta64(2, 'Y')), 0, 1) == 1:
            df.loc[df['ACCOUNT NUMBER'] == account, 'exists_2_yrs'] = 1
        else:
            df.loc[df['ACCOUNT NUMBER'] == account, 'exists_2_yrs'] = 0   


def process_blocks():
    '''

    '''
    df = pd.read_csv('blocks.csv')
    df['GEOID10'] = df['GEOID10'].astype(str)
    df["block_group"] = df["GEOID10"].apply(lambda block: block[:12])
    df["the_geom"] = df["the_geom"].apply(shapely.wkt.loads)
    gdf = gpd.GeoDataFrame(df).set_geometry("the_geom").drop(columns=df.columns.difference(["block_group", "the_geom"]))
    gdf = gpd.GeoDataFrame(gdf).set_geometry('the_geom')
    return gdf


def join_with_block_groups(business_df, blocks_df):
    '''
    '''
    business_df = business_df.dropna(subset=["LONGITUDE", "LATITUDE"])
    business_df["the_geom"] = business_df.apply(lambda row: Point(float(row["LONGITUDE"]), float(row["LATITUDE"])), axis=1)
    business_gdf = gpd.GeoDataFrame(business_df).set_geometry("the_geom")
    joined_data = gpd.sjoin(business_gdf, blocks_df[["block_group", "the_geom"]]).drop(columns="index_right")
    return joined_data

def find_num_bus(df):
    '''
    Finds current number of businesses within a block group
    '''
    df["num_open_bus"] = np.nan
    for group in df['block_group'].unique():
        cnt = df[(df['block_group'] == group)]['status'].value_counts()
        df.loc[df['block_group'] ==  group, 'num_open_bus'] = cnt[1]


COLUMN_NAMES = ['APPLICATION TYPE',  'LICENSE DESCRIPTION']
def breakdown_by_blck_grp(df, col):
    '''
    Finds percentage breakdown for each type of a given column for each block group 
    e.g. what's the percentage of renewals in this blk group?
    '''
    val_lst = list(df[col].unique())
    grouped = df.groupby('block_group')[col].value_counts().unstack(fill_value=0)
    pct = grouped[val_lst].div(grouped.sum(axis=1), axis=0)*100
    return pct.reset_index()


# do we want to include anything not issue/renew?
# we could also do axis=0 for the .sum to get a comparison across block groups
