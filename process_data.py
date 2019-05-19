import geopandas as gpd
import numpy as np
from shapely.geometry import shape, Point
import shapely.wkt
import pandas as pd
from dateutil.relativedelta import relativedelta


def process_census(acs_csv):
    '''
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


def process_business():
    '''
    '''
    df = pd.read_csv('business.csv')
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col])
    to_drop_lst = []
    for col in df.columns:
      if "computed" in col:
        to_drop_lst.append(col)
    df.drop(to_drop_lst, axis=1, inplace=True)
    change_date_type(df)
    find_duration(df, 'DATE ISSUED', 'APPLICATION REQUIREMENTS COMPLETE')
    return df


def change_date_type(dataframe):
    '''
    Converts columns with dates to datetime objects

    Inputs: a pandas dataframe

    Outputs: None
    '''
    for col in dataframe.columns: 
        if "DATE" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])
        if col in ['APPLICATION CREATED DATE', 
                   'APPLICATION REQUIREMENTS COMPLETE',
                   'LICENSE APPROVED FOR ISSUANCE']:
            dataframe[col] = pd.to_datetime(dataframe[col])


def find_duration(df, col1, col2):
    '''
    Calculates the amount of days that have passed between two columns
    '''
    diff = df[col2] - df[col1]
    df['days_between_' + col1 + "_" + col2] = diff.dt.days


def check_alive(df, date):
    '''
    Checks to see if business still open after given date
    0: closed
    1: open

    Outputs: None
    '''
    for account in df['ACCOUNT NUMBER'].unique():
        df['status'] = np.where(df[df['ACCOUNT NUMBER'] == account]['LICENSE TERM EXPIRATION DATE'].max() < date, 1, 0)


def make_dummy_cat(dataframe, col_lst):
    '''
    Creates new columns of dummy variables from categorical columns of the data

    Inputs:
        dataframe: a pandas dataframe
        col_lst: list of columns to convert to dummy columns

    Outputs: a pandas dataframe
    '''
    dfs_to_concat = [dataframe]
    for column in col_lst:
        dummy_df = pd.get_dummies(dataframe[column], prefix=column)
        dfs_to_concat.append(dummy_df)
    dataframe = pd.concat(dfs_to_concat, axis=1)
    for column in col_lst:
        dataframe.drop(column, axis=1, inplace=True)
    return dataframe


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


def working(business_df, blocks_df, census_df):
    '''
    '''
    joined = join_with_block_groups(business_df, blocks_df)
    full = pd.merge(joined, census_df, on='block_group')


def find_num_bus(df):
    '''
    Finds current number of businesses within a block group
    '''
    df.groupby('block_group').count('status')


'''    
new_bus_dict = {}
new_bus_dict['type'] = ['Percentage of New Applications']
prev_year = 2002
for year in year_lst:
    last_year = set(augmented_df[(augmented_df['date_issued'].dt.year == prev_year) & (augmented_df['application_type'] == 'ISSUE')]['account_number'])
    this_year = set(augmented_df[(augmented_df['date_issued'].dt.year == year) & (augmented_df['application_type'] == 'ISSUE')]['account_number'])
    diff = len(this_year) - len(last_year)
    new_bus_dict[year] = [diff*100 / len(last_year)]
    prev_year = year

new_bus = pd.DataFrame.from_dict(new_bus_dict)

renewals_dict = {}
renewals_dict['type'] = ['Percentage of Renewals']
prev_year = 2002
for year in year_lst:
    last_year = set(augmented_df[(augmented_df['date_issued'].dt.year == prev_year) & (augmented_df['application_type'] == 'RENEW')]['account_number'])
    this_year = set(augmented_df[(augmented_df['date_issued'].dt.year == year) & (augmented_df['application_type'] == 'RENEW')]['account_number'])
    diff = len(this_year) - len(last_year)
    renewals_dict[year] = [diff*100 / len(last_year)]
    prev_year = year

renewals = pd.DataFrame.from_dict(renewals_dict)
'''