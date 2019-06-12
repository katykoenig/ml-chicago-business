import process_data as pr
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
                         'logistic_regression', 'bagging']
    elif model == 'all':
        models_to_run = ['decision_tree', 'random_forest', 'knn',
                         'logistic_regression', 'ada_boost',
                         'gradient_boost', 'bagging']
    else:
        models_to_run = model

    for pair in DATA_LST:
        train_df = pd.read_csv(pair[0])
        test_df = pd.read_csv(pair[1])
        train_df = pr.clean_types(train_df)
        test_df = pr.clean_types(test_df)
        train_df, test_df = pr.add_crimes(train_df, test_df, pair[0], pair[1])
        features_lst, train_df = pl.generate_features(train_df, ['successful','earliest_issue', 'latest_issue','sf_id', 'block'])
        thresh_lst = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        results_df = pl.combining_function2(features_lst, models_to_run, thresh_lst, 'successful', train_df, test_df)
        print('writing results for ' + pair[1])
        results_df.to_csv(pair[1][5:-4] + str(pd.Timestamp.now()) + '.csv')
        eval_lst = ['auc_roc']
        pl.results_eval(results_df, eval_lst, train_df, test_df, 'successful', features_lst, pair[1][5:-4])
        pl.print_feature_importance(train_df[features_lst], train_df['successful'])


#go.eval_results('predscores.csv', 0.1, 'true', 'scores')

def eval_results(results_file, threshold_for_tracts, y_true_col, y_score_col):
    results = pd.read_csv(results_file)
    results = results.dropna(axis=0)
    y_scores_sorted, y_true_sorted = pl.joint_sort_descending(
        np.array(results[y_score_col]), np.array(results[y_true_col]))
    #create precision recall and AUC graphs
    pl.plot_precision_recall_n(y_true_sorted, y_scores_sorted, '','','')

    #get best census tracts
    avg_score_by_tract = results.groupby('census_tract').sum()[y_score_col] / results.groupby('census_tract').size()
    avg_score_by_tract = avg_score_by_tract.sort_values(ascending=False)
    cutoff = int(len(avg_score_by_tract) * threshold_for_tracts)
    print(len(avg_score_by_tract))
    return avg_score_by_tract.iloc[:cutoff]


from shapely.ops import unary_union
import requests
import geopandas as gpd
from shapely.geometry import shape, Point
import matplotlib.pyplot as plt

def make_chicago_map(file_name):
    avg_score_by_tract = pd.read_csv(file_name, names=['census_tract', 'score'])
    avg_score_by_tract['census_tract'] = avg_score_by_tract['census_tract'].astype('str')
    os = 0
    params = {'$limit': 1000, '$offset': os}
    response = requests.get('https://data.cityofchicago.org/resource/74p9-q2aq.json', params).json()
    total_response = response
    os = 1000
    while len(response) >= 1000:
        params = {'$limit': 1000, '$offset': os}
        response = requests.get("https://data.cityofchicago.org/resource/74p9-q2aq.json", params).json()
        os += 1000
        total_response += response
    tracts = pd.DataFrame.from_dict(total_response)
    tracts["the_geom"] = tracts["the_geom"].apply(shape).apply(unary_union)
    tracts = gpd.GeoDataFrame(tracts).set_geometry("the_geom").drop(columns=tracts.columns.difference(["geoid10", "the_geom"]))
    tracts_gdf = gpd.GeoDataFrame(tracts).set_geometry('the_geom')

    merged = pd.merge(avg_score_by_tract, tracts_gdf, left_on='census_tract', right_on='geoid10', how='outer')
    merged = merged.drop('census_tract', axis=1)
    merged = merged.fillna(0)
    merged = merged.dropna()

    merged['score'] = pd.qcut(merged['score'], 10, labels=[1,2,3,4,5,6,7,8,9,10])
    merged_gdf = gpd.GeoDataFrame(merged, geometry='the_geom')


    fig, ax = plt.subplots(figsize=(15, 15))
    chicagomap = merged_gdf.plot(ax=ax, column='score', legend=True)
    chicagomap.figure.savefig('map.png')





