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
    print("Summary Statistics")
    print(joined.describe())
    print()
    # 3 calls below save images of some stats from df
    ds.evaluate_correlations(joined, 'correlation_heatmap.png')
    ds.show_distribution(joined)
    ds.create_scatterplots(joined)
    print('Summary of Null Values')
    print(joined.isnull().sum(axis=0))
    print()
    for col in joined.columns:
        if 'date' in col:
            joined[col] = pd.to_datetime(joined[col])
    train_start = joined['earliest_date_issued'].min()
    test_end = joined['latest_date_issued'].max()
    test_start = test_end - relativedelta(months=+24, days=+1)
    train_end = test_end - relativedelta(months=+24)
    print(train_start, train_end)
    print(test_start, test_end)
    # the 2 fn calls below are to find how many living neighbor businesses each bus has (a feature)
    print('finding alive neighbors')
    pr.check_alive(joined, train_end)
    pr.find_alive_neighbors(joined)
    # need to get col names for feature list below
    return joined
    #features_lst = []
    #pl.split_data(joined, features_lst, data)

