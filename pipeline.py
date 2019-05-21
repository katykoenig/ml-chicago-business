'''
Kathryn (Katy) Koenig
CAPP 30254

Functions for Creating ML Pipeline
'''
import csv
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score as accuracy, precision_recall_curve
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from dateutil.relativedelta import relativedelta


def temporal_validate(dataframe, col, window):
    '''
    Creates list of for start and end dates for training and testing sets of the data.

    Inputs:
        dataframe: a pandas dataframe
        col: date column
        window (integer): length of time for which we are predicting

    Outputs:
        dates_lst:
    '''
    train_start_time = dataframe[col].min()
    end_time = dataframe[col].max()
    train_end_time = train_start_time + relativedelta(months=+window, days=-1)
    test_start_time = train_start_time + relativedelta(months=+window)
    test_end_time = test_start_time + relativedelta(months=+window)
    dates_lst = [(train_start_time, train_end_time, test_start_time, test_end_time)]
    while end_time >= test_start_time + relativedelta(months=+window):
        train_end_time = train_end_time + relativedelta(months=+window)
        test_start_time += relativedelta(months=+window)
        test_end_time = test_start_time + relativedelta(months=+window)
        dates_lst.append((train_start_time, train_end_time, test_start_time, test_end_time))
    return dates_lst


def split_data(dataframe, date_col, features_lst, date):
    '''
    Splits data into testing and training datasets

    Inputs:
        dataframe: a pandas dataframe
        features_lst: list of column names of features/predictors
        target_att: outcome variable to be prediced (a column name)
        split_size:

    Output:
        x_train: pandas dataframe with only features columns of training data
        x_test: pandas dataframe with only features columns of testing data
        y_train: pandas series with outcome column of training data
        y_test: pandas seires with outcome column of testing data
    '''
    train_start_time = date[0]
    train_end_time = date[1]
    test_start_time = date[2]
    test_end_time = date[3]
    training_df = dataframe[(dataframe[date_col] >= train_start_time)
                            & (dataframe[date_col] <= train_end_time)]
    x_train = training_df[features_lst]
    survive_two_years(training_df)
    y_train = training_df['exists_2_yrs']
    testing_df = dataframe[(dataframe[date_col] >= test_start_time) & (dataframe[date_col] <= test_end_time)]
    x_test = testing_df[features_lst]
    survive_two_years(testing_df)
    y_test = testing_df['exists_2_yrs']
    return x_train, x_test, y_train, y_test


def survive_two_years(df):
    '''
    '''
    df['exists_2_yrs'] = df['days_alive'] < 731
    df['exists_2_yrs'] = df['exists_2_yrs'].astype(int)


def discretize_variable_by_quintile(dataframe, col_name):
    '''
    Discretizes and relabels values in a column by breaking it into quintiles

    Inputs:
        dataframe: a pandas dataframe
        col_name: name of column to be discretized into quintiles

    Outputs: a pandas dataframe
    '''
    dataframe[col_name] = pd.qcut(dataframe[col_name], 5, labels=[1, 2, 3, 4, 5])


def discretize_by_unique_val(dataframe, col_lst):
    '''
    Discretizes categorical columns in col_lst to integer values

     Inputs:
        dataframe: a pandas dataframe
        col_lst: list of column names to be discretized

    Ouptuts:
        master_dict: a dictionary with the column names, mapping the integer
                     values to their meanings
        dataframe: a pandas dataframe
    '''
    master_dict = {}
    for col in col_lst:
        discret_dict = {}
        counter = 0
        for i in dataframe[col].unique():
            discret_dict[i] = counter
            counter += 1
        dataframe[col] = dataframe[col].map(discret_dict)
        master_dict[col] = discret_dict
    return master_dict, dataframe


def discretize_dates(dataframe, features_lst):
    '''
    Converts datetime types into integer of month

    Inputs:
        dataframe: a pandas dataframe
        features_lst: a list of columns

    Outputs: None
    '''
    types_df = dataframe.dtypes.reset_index()
    datetime_df = types_df[types_df[0] == 'datetime64[ns]']
    to_discretize = list(datetime_df['index'])
    for col in to_discretize:
        dataframe["month_" + col[-6:]] = dataframe[col].dt.month
        features_lst.append("month_" + col[-6:])
    return features_lst, dataframe


def make_dummy_cont(dataframe, column, desired_col_name, cutoff):
    '''
    Creates new column of dummy variables where the value becomes 1 if above a
    given cutoff point and 0 if below cutoff point and drops original column

    Inputs:
        dataframe: a pandas dataframe
        column: name of column to be converted to dummy variable column
        desired_col_name: new column name for dummy variable column
        cutoff: cutoff point for which new column value becomes 1 if above and 0 if below.

    Outputs: None
    '''
    dataframe[desired_col_name] = np.where(dataframe[column] <= cutoff, 1, 0)
    dataframe.drop(column, axis=1, inplace=True)


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


def generate_features(dataframe, target_att, drop_lst):
    '''
    Generates the list of features/predictors to be used in training model

    Inputs:
        dataframe: a pandas dataframe
        target_att: outcome variable to be prediced (a column name)
        drop_lst: list of columns to not be included in features

    Output:
        features_lst: list of column names of features/predictors
    '''
    features_lst = [i for i in list(dataframe.columns) if i != target_att
                    if "id" not in i if 'date' not in i if i not in drop_lst]
    return features_lst

# The code below relies on Rayid Ghani's magic loop, found here:
# https://github.com/rayidghani/magicloops

params_dict = { 
    'random_forest': {'n_estimators': [10, 100, 1000], 'max_depth': [1, 5, 10],
                      'min_samples_split': [2, 5, 10], 'n_jobs': [-1]},
    'logistic_regression': {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10],
                            'solver': ['lbfgs']},
    'decision_tree': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10],
                      'min_samples_split': [2, 5, 10]},
    'knn': {'n_neighbors': [5, 10, 25, 50], 'weights': ['uniform', 'distance']},
    'ada_boost': {'algorithm': ['SAMME.R'],
                  'n_estimators': [1, 10, 100, 1000]},
    'gradient_boost': {'n_estimators': [10, 100], 'max_depth': [3, 5, 10]},
    'bagging': {'n_estimators': [10, 100], 'random_state': [0]}}

clfs = {'random_forest': RandomForestClassifier(),
        'logistic_regression': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(),
        'knn': KNeighborsClassifier(),
        'ada_boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),
        'gradient_boost': GradientBoostingClassifier(),
        'bagging': BaggingClassifier(DecisionTreeClassifier(max_depth=5))}

results_col = ['model', 'parameters', 'train_start', 'train_end', 'test_start',
               'test_end', 'test_baseline', 'accuracy_at_1', 'precision_at_1',
               'recall_at_1', 'f1_score_at_1', 'auc_roc_at_1', 'accuracy_at_2',
               'precision_at_2', 'recall_at_2', 'f1_score_at_2',
               'auc_roc_at_2', 'accuracy_at_5', 'precision_at_5',
               'recall_at_5', 'f1_score_at_5', 'auc_roc_at_5',
               'accuracy_at_10', 'precision_at_10', 'recall_at_10',
               'f1_score_at_10', 'auc_roc_at_10', 'accuracy_at_20',
               'precision_at_20', 'recall_at_20', 'f1_score_at_20',
               'auc_roc_at_20', 'accuracy_at_30', 'precision_at_30',
               'recall_at_1', 'f1_score_at_30', 'auc_roc_at_30',
               'accuracy_at_50', 'precision_at_50', 'recall_at_50',
               'f1_score_at_50', 'auc_roc_at_50']


def combining_function(outputfile, date, model_lst, dataframe, col, features_lst):
    '''
    Creates models, evaluates models and writes evaluation of models to csv.

    Input:
        outputfile: csv filename
        date_lst: a list of dates on which to split training and testing data
        model_lst: list of classifier models to run
        dataframe: a pandas dataframe
        col: target column for prediction
        features_lst: list of columns to be considered for features in model

    Outputs:
    '''
    with open(outputfile, 'w') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        outputwriter.writerow(results_col)
        x_train, x_test, y_train, y_test = split_data(dataframe, col, features_lst, date)
        train_start = date[0]
        train_end = date[1]
        test_start = date[2]
        test_end = date[3]
        for model in model_lst:
            clf = clfs[model]
            params_to_run = params_dict[model]
            for p in ParameterGrid(params_to_run):
                row_lst = [model, p, train_start, train_end, test_start,
                           test_end, np.mean(y_test)]
                clf.set_params(**p)
                clf.fit(x_train, y_train)
                predicted_scores_test = clf.predict_proba(x_test)[:, 1]
                total_lst = []
                for threshold in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
                    calc_threshold = lambda x, y: 0 if x < y else 1
                    predicted_test = np.array([calc_threshold(score, threshold)
                                               for score in predicted_scores_test])
                    acc = accuracy(y_pred=predicted_test, y_true=y_test)
                    prec = precision_score(y_pred=predicted_test, y_true=y_test)
                    recall = recall_score(y_pred=predicted_test, y_true=y_test)
                    f_one = f1_score(y_pred=predicted_test, y_true=y_test)
                    auc_roc = roc_auc_score(y_score=predicted_test, y_true=y_test)
                    spec_results_lst = [acc, prec, recall, f_one, auc_roc]
                    total_lst += spec_results_lst
                outputwriter.writerow(row_lst + total_lst)
    csvfile.close()


#To understand plotting precision-recall and ROC curves, this work was
# informed by the following site:
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

def create_curves(model, params, x_train, y_train, x_test, y_test):
    '''
    Prints area under the curve and creates and saves an ROC and precision-recall curves image

    Inputs:
        model: name of machine learning classifer
        params: params for classifier to run
        x_train: pandas dataframe with only features columns of training data
        x_test: pandas dataframe with only features columns of testing data
        y_train: pandas series with outcome column of training data
        y_test: pandas seires with outcome column of testing data

    Outputs: None
    '''
    clf = clfs[model]
    params = ast.literal_eval(params)
    clf.set_params(**params)
    clf.fit(x_train, y_train)
    predicted_scores_test = clf.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, predicted_scores_test)
    print(model)
    print('AUC: %.3f' % auc)
    fpr, tpr, _ = roc_curve(y_test, predicted_scores_test)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    roc_title = "ROC " + model + " with " + str(params)
    plt.title(roc_title)
    plt.savefig(roc_title + '.png')
    plt.clf()

    precision, recall, _ = precision_recall_curve(y_test, predicted_scores_test)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    plt.plot(recall, precision, marker='.')
    p_r_title = "Precision-Recall " + model + " with " + str(params)
    plt.title(p_r_title)
    plt.savefig(p_r_title + '.png')
    plt.clf()


def results_eval(csvfile, dataframe, col, features_lst, dates):
    '''
    Evaluates the results of the models run and creates AUC-ROC and
    precision-recall curves for models deemed best

    Inputs:
        csvfile: name of results csv file
        dataframe: a pandas dataframe
        col: target column for prediction
        features_lst: list of columns to be considered for features in model
        dates: a list of dates on which to split training and testing data
        model_lst: list of classifier models to run

    Outputs: None
    '''
    evaluator_lst = ['accuracy_at_5', 'precision_at_5', 'recall_at_5',
                     'f1_score_at_5', 'auc_roc_at_5']
    results_df = pd.read_csv(csvfile)
    for date in dates:
        print("BEST MODELS FOR START TEST DATE" + str(date[2]))
        x_train, x_test, y_train, y_test = split_data(dataframe, col, features_lst, date)
        test_start = str(date[2])
        specified_df = results_df[results_df['test_start'] == test_start]
        for evaluator in evaluator_lst:
            best_index = specified_df[evaluator].idxmax()
            best_mod = results_df.iloc[best_index, 0:2]
            print(best_mod)
            print(results_df.iloc[best_index, 17:22])
            print()
            create_curves(best_mod[0], best_mod[1], x_train, y_train, x_test, y_test)