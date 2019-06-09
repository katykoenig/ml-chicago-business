# import process_data as pr
import pandas as pd
import numpy as np
import pipeline as pl
#import descriptive_stats as ds
from dateutil.relativedelta import relativedelta

DATA_LST = [('data/train_2010-01-01_2012-06-01.csv', 'data/valid_2014-05-31_2014-06-01.csv'),
            ('data/train_2010-01-01_2013-06-01.csv', 'data/valid_2015-05-31_2015-06-01.csv'),
            ('data/train_2010-01-01_2014-06-01.csv', 'data/valid_2016-05-31_2016-06-01.csv'),
            ('data/train_2010-01-01_2015-06-01.csv', 'data/valid_2017-05-31_2017-06-01.csv')]

def go(model=None):
    '''
    '''
    if not model:
        models_to_run = ['decision_tree', 'random_forest', 'knn',
                         'logistic_regression', 'ada_boost', 'bagging']
    elif model == 'all':
        models_to_run = ['decision_tree', 'random_forest', 'knn',
                         'logistic_regression', 'ada_boost',
                         'gradient_boost', 'bagging']
    else:
        models_to_run = model

    for pair in DATA_LST:
        train_df = pd.read_csv(pair[0])
        test_df = pd.read_csv(pair[1])
        # train_df = pr.clean_types(train_df)
        # test_df = pr.clean_types(test_df)
        features_lst, train_df = pl.generate_features(test_df, ['successful','earliest_issue', 'latest_issue','sf_id', 'block_group'])
        thresh_lst = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        results_df = pl.combining_function2(features_lst, models_to_run, thresh_lst, 'successful', train_df, test_df)
        print('writing results for ' + pair[1])
        results_df.to_csv('results_df ' + pair[1])
        eval_lst = ['accuracy_at_5', 'precision_at_5', 'recall_at_5', 'f1_score_at_5', 'auc_roc_at_5']
        pl.results_eval(results_df, eval_lst, train_df, test_df, 'successful', features_lst, pair[1])
