import process_data as pr
import pandas as pd
import numpy as np
import pipeline as pl
import descriptive_stats as ds
from dateutil.relativedelta import relativedelta


def go():
    '''
    '''
    blocks = pr.process_blocks()
    train_df = pd.read_csv('train_test_sets/train_set_20170531.csv')
    train_df = pr.clean_types(train_df)
    test_df = pd.read_csv('test_set_20170531.csv')
    test_df = pr.clean_types(test_df)
    #this joins blocks and crimes
    train = pr.process_crime(blocks, train_df)
    test = pr.process_crime(blocks, test_df)
    features_lst = pl.generate_features(test, 'successful', ['late_expiry'])
    results_df = pl.combining_function('modelresults.csv', date, model_lst, joined, 'latest_date_issued', features_lst)
    results_df.to_csv('results.csv')
