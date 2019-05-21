import process_business as pr
import pandas as pd
import numpy as np
import pipeline as pl
import descriptive_stats as ds


def go():
	'''
	'''
    bus_df = pr.process_business()
    pr.make_bus(bus_df)
    cleaned_bus = pd.read_csv('cleanedbus.csv')
    blocks_df = pr.process_blocks()
    acs_df = pr.process_census()
    merged = pd.merge(acs_df, blocks_df, left_on='block_group', right_on='GEOID10')
    joined = join_with_block_groups(cleaned_bus, merged)
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
    dates_lst = pl.temporal_validate(joined, 'earliest_date_issued', 24)
    # using most recent (aka largest train, most recent test)
    date = dates_lst[-1]
    train_start = date[0]
    train_end = date[1]
    test_start = date[2]
    test_end = date[3]
    # the 2 fn calls below are to find how many living neighbor businesses each bus has (a feature)
    pr.check_alive(joined, train_end)
    pr.find_alive_neighbors(joined)
    # need to get col names for feature list below
    features_lst = []
    pl.split_data(joined, features_lst, data)