import process_data as pr
import pandas as pd
import numpy as np
import pipeline as pl
import descriptive_stats as ds
from dateutil.relativedelta import relativedelta


def go():
    '''
    '''
    #bus_df = pr.process_business()
    #pr.make_bus(bus_df)
    cleaned_bus = pd.read_csv('cleanedbus.csv')
    blocks_df = pr.process_blocks()
    acs_df = pr.process_census()
    merged = pd.merge(acs_df, blocks_df, on='block_group')
    joined = pr.join_with_block_groups(cleaned_bus, merged)
    for col in joined.columns:
        if 'date' in col:
            joined[col] = pd.to_datetime(joined[col])
    pr.length_alive(joined)
    print("Summary Statistics")
    print(joined.describe())
    print()
    # 2 calls below save images of some stats from df
    ds.evaluate_correlations(joined, 'correlation_heatmap.png')
    ds.show_distribution(joined)
    print('Summary of Null Values')
    print(joined.isnull().sum(axis=0))
    print()
    train_start = joined['earliest_date_issued'].min()
    test_end = joined['latest_date_issued'].max()
    test_start = test_end - relativedelta(months=+24, days=+1)
    train_end = test_end - relativedelta(months=+24)
    date = (train_start, train_end, test_start, test_end)
    print(train_start, train_end)
    print(test_start, test_end)
    # the 2 fn calls below are to find how many living neighbor businesses each bus has (a feature)
    print('finding alive neighbors')
    pr.check_alive(joined, train_end)
    pr.find_alive_neighbors(joined)
    # need to recalc that based on split by dates 
    joined.drop(columns = 'exists_2_yrs', inplace=True)

    features_lst = ['1010', '1006', '1781', '1475', '4404', '1012', '1011',
                    '1470', '1474', '1009', '1329', '1008', '1569', '1007',
                    '8340', '1003', '1050', '1625', '1604', '4406',
                    'pct_male_children', 'pct_male_working',
                    'pct_male_elderly', 'pct_female_chilren',
                    'pct_female_working', 'pct_female_elderly', 
                    'pct_low_travel_time', 'pct_med_travel_time',
                    'pct_high_travel_time', 'pct_below_pov', 'pct_below_med',
                    'pct_above_med', 'pct_high_inc', 'pct_race_white',
                    'pct_race_black', 'pct_race_asian', 'days_alive']
    models_lst = ['decision_tree', 'random_forest', 'knn', 'logistic_regression', 'ada_boost', 'bagging']
    pl.combining_function('modelresults.csv', date, model_lst, joined, 'latest_date_issued', features_lst)

    #when evaluating look at which models perform best for demos
    #but also ONLY FOR ONES WHERE STATUS == 1 (aka is an open business) - who cares about a bus that's already closed

