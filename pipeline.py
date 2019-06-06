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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score as accuracy, precision_recall_curve
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from dateutil.relativedelta import relativedelta
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image


def discretize_dates(dataframe, features_lst):
    '''
    Converts datetime types into integer of month and adds new discretized
    date columns to features list
    Inputs:
        dataframe: a pandas dataframe
        features_lst: a list of columns
    Outputs:
        features_lst: a list of updated columns
        dataframe: updated pandas dataframe
    '''
    types_df = dataframe.dtypes.reset_index()
    datetime_df = types_df[types_df[0] == 'datetime64[ns]']
    to_discretize = list(datetime_df['index'])
    for col in to_discretize:
        new_col = "month" + col[-6:]
        dataframe[new_col] = dataframe[col].dt.month
        if new_col not in features_lst:
            features_lst.append("month" + col[-6:])
    return features_lst, dataframe


def generate_features(dataframe, drop_lst):
    '''
    Generates the list of features/predictors to be used in training model
    Inputs:
        dataframe: a pandas dataframe
        target_att: outcome variable to be prediced (a column name)
        drop_lst: list of columns to not be included in features
    Output:
        features_lst: list of column names of features/predictors
    '''
    features_lst = [i for i in list(dataframe.columns) if i not in drop_lst]
    return discretize_dates(dataframe, features_lst)


# The code below relies on Rayid Ghani's magic loop, found here:
# https://github.com/rayidghani/magicloops

def generate_binary_at_k(y_scores, k):
    '''
    Converts classifier predictions to binary based on desired
    percentage/threshold
    Inputs:
        y_scores: a series of probability prediction made by classifier
        k: a float, denoting the threshold
    Outputs: a pandas series of binary values
    '''
    cutoff_index = int(len(y_scores) * k)
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def joint_sort_descending(array_one, array_two):
    '''
    Sorts two arrays in descending order
    Inputs:
        array_one: a numpy array
        array_two: a numpy array
    Outputs: two sorted arrays
    '''
    idx = np.argsort(array_one)[::-1]
    return array_one[idx], array_two[idx]


PARAMS_DICT = {
    'random_forest': {'n_estimators': [10, 100, 500], 'max_depth': [1, 5, 10],
                      'min_samples_split': [2, 5, 10], 'n_jobs': [-1]},
    'logistic_regression': {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10],
                            'solver': ['lbfgs']},
    'decision_tree': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10],
                      'min_samples_split': [2, 5, 50]},
    'SVM': {},
    'knn': {'n_neighbors': [5, 10, 25, 50], 'weights': ['uniform', 'distance']},
    'ada_boost': {'algorithm': ['SAMME.R'],
                  'n_estimators': [1, 10, 100, 1000]},
    'gradient_boost': {'n_estimators': [10, 100], 'max_depth': [3, 5, 10]},
    'bagging': {'n_estimators': [10, 100], 'random_state': [0], 'n_jobs': [-1]}}

CLFS = {'random_forest': RandomForestClassifier(),
        'logistic_regression': LogisticRegression(),
        'SVM': svm.SVC(random_state=0, probability=True),
        'decision_tree': DecisionTreeClassifier(),
        'knn': KNeighborsClassifier(),
        'ada_boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),
        'gradient_boost': GradientBoostingClassifier(),
        'bagging': BaggingClassifier(DecisionTreeClassifier(max_depth=5))}

RESULTS_COLS = ['model', 'parameters', 'train_start', 'train_end', 'test_start',
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


def combining_function(features_lst, model_lst, threshold_lst, target_att, train_df, test_df):
    '''
    Creates models, evaluates models and writes evaluation of models to csv.
    Input:
        date_lst: a list of dates on which to split training and testing data
        model_lst: list of classifier models to run
        dataframe: a pandas dataframe
        col: target column for prediction
        dummy_lst: list of column names to be converted to dummy variables
        discretize_lst: list of column names to be discretized
        threshold_lst: list of threshold values
        target_att: outcome variable to be prediced (a column name)
        drop_lst: list of column names to not be considered features
    Outputs: a pandas dataframe with the results of our models
    '''
    results_df = pd.DataFrame(columns=RESULTS_COLS)
    train_start = train_df['earliest_issue'].min()
    train_end = train_df['earliest_issue'].max()
    test_start = test_df['earliest_issue'].min()
    test_end = test_df['earliest_issue'].max()
    _, test_df = discretize_dates(test_df, features_lst)
    x_train = train_df[features_lst]
    y_train = train_df[target_att]
    x_test = test_df[features_lst]
    y_test = test_df[target_att]
    # Loop through models and differing parameters
    # while fitting each model with split data
    for model in model_lst:
        print('Running model ' + model + ' for test start date ' + str(test_start))
        clf = CLFS[model]
        params_to_run = PARAMS_DICT[model]
        # Loop through varying paramets for each model
        for param in ParameterGrid(params_to_run):
            row_lst = [model, param, train_start, train_end, test_start,
                       test_end, np.mean(y_test)]
            clf.set_params(**param)
            clf.fit(x_train, y_train)
            predicted_scores = clf.predict_proba(x_test)[:, 1]
            total_lst = []
            # Loop through thresholds,
            # and generating evaluation metrics for each model
            for threshold in threshold_lst:
                y_scores_sorted, y_true_sorted = joint_sort_descending(
                    np.array(predicted_scores), np.array(y_test))
                preds_at_k = generate_binary_at_k(y_scores_sorted,
                                                  threshold)
                acc = accuracy(y_true_sorted, preds_at_k)
                prec = precision_score(y_true_sorted, preds_at_k)
                recall = recall_score(y_true_sorted, preds_at_k)
                f_one = f1_score(y_true_sorted, preds_at_k)
                auc_roc = roc_auc_score(y_true_sorted, preds_at_k)
                total_lst += [acc, prec, recall, f_one, auc_roc]
            results_df.loc[len(results_df)] = row_lst + total_lst
    return results_df


#To understand plotting the AUC-ROC curve, this work was informed by the
#following site:
#https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

def results_eval(dataframe, evaluator_lst, train_df, test_df, target_att, features_lst, date):
    '''
    Evaluates the results of the models run and creates AUC-ROC and
    precision-recall curves for models deemed best
    Inputs:
        dataframe: a pandas dataframe
        results_df:
        col: target column for prediction
        dates: a list of dates on which to split training and testing data
        dummy_lst: list of column names to be converted to dummy variables
        discretize_lst: list of column names to be discretized
        outcome variable to be prediced (a column name)
        drop_lst: list of column names to not be considered features
        evaluator_lst: list of evaluation metrics
    Outputs: None
    '''

    _, test_df = discretize_dates(test_df, features_lst)
    x_train = train_df[features_lst]
    y_train = train_df[target_att]
    x_test = test_df[features_lst]
    y_test = test_df[target_att]
    for i in results_df['test_set'].unique():
        for evaluator in evaluator_lst:
            print("BEST MODEL FOR " + str(i) + ' with '+ evaluator)
            best_index = specified_df[evaluator].idxmax()
            best_mod = results_df.iloc[best_index, 0:2]
            print(best_mod)
            print(results_df.iloc[best_index, 17:22])
            print()
            if i == 'full':
                create_curves(best_mod[0], best_mod[1], x_train, y_train, x_test, y_test, date)


def create_curves(model, params, x_train, y_train, x_test, y_test, threshold=.05):
    '''
    Prints area under the curve and creates and saves an ROC and precision-recall curves image
    Inputs:
        model: name of machine learning classifer
        params: params for classifier to run
        x_train: pandas dataframe with only features columns of training data
        x_test: pandas dataframe with only features columns of testing data
        y_train: pandas series with outcome column of training data
        y_test: pandas series with outcome column of testing data
    Outputs: None
    '''
    clf = CLFS[model]
    clf.set_params(**params)
    clf.fit(x_train, y_train)
    predicted_scores = clf.predict_proba(x_test)[:, 1]
    y_scores_sorted, y_true_sorted = joint_sort_descending(
        np.array(predicted_scores), np.array(y_test))
    preds_at_k = generate_binary_at_k(y_scores_sorted, threshold)
    auc = roc_auc_score(y_true_sorted, preds_at_k)
    print(model)
    print('AUC: %.3f' % auc)
    fpr, tpr, _ = roc_curve(y_true_sorted, preds_at_k)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    roc_title = "ROC " + model + " with " + str(params) + date
    plt.title(roc_title)
    plt.savefig(roc_title + '.png')
    plt.clf()
    plot_precision_recall_n(y_true_sorted, preds_at_k, model, params)


# The code below also comes from Rayid Ghani's magic loop, again found here:
# https://github.com/rayidghani/magicloops

def plot_precision_recall_n(y_true, y_score, model, params):
    '''
    Plots and saves precision-recall curve for a given model
    Inputs:
        y_true: pandas series with outcome column
        y_score: pandas series of predicted outcome
        model: name of machine learning classifer
        params: params for classifier to run
    Outputs: None
    '''
    precision_curve, recall_curve, pr_thresh = precision_recall_curve(y_true,
                                                                      y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresh:
        num_above_thresh = np.count_nonzero([y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    _, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])
    p_r_title = "Precision-Recall " + model + " with " + str(params) + date
    plt.title(p_r_title)
    plt.savefig(p_r_title + '.png')
    plt.clf()


def rep_d_tree(dec_tree, features_lst):
    '''
    Saves a .png representation of the decision tree
    Input: decision tree object
    Outputs: None
    '''
    dot_data = StringIO()
    export_graphviz(dec_tree, feature_names=features_lst, out_file=dot_data,
                    filled=True, rounded=True, special_characters=True,)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.write_png('decision_tree.png'))