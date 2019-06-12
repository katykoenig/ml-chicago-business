'''
Promoting Sustained Entrepreneurship in Chicago
Machine Learning for Public Policy
University of Chicago, CS & Harris School of Public Policy
June 2019

Rayid Ghani (@rayidghani)
Katy Koenig (@katykoenig)
Eric Langowski (@erhla)
Patrick Lavallee Delgado (@lavalleedelgado)

This module is based on a machine learning pipeline developed by Katy Koenig.

'''

import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, \
    precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sqlite3
from db_query import temporal_select


TEMPORAL_SPLITS = [
    ("2010-01-01", "2015-06-01", "2017-06-01", "2017-06-02"),
    ("2010-01-01", "2014-06-01", "2016-06-01", "2016-06-02"),
    ("2010-01-01", "2013-06-01", "2015-06-01", "2015-06-02"),
    ("2010-01-01", "2012-06-01", "2014-06-01", "2014-06-02")
]
TARGET_VARIABLE = "successful"
RESERVED_COLUMNS = [
    "sf_id",
    "block",
    "successful",
    "earliest_issue",
    "latest_issue"
]
DROP_COLUMN_KEYWORDS = ["crime", "license"]
THRESHOLDS = [1, 2, 5, 10, 20, 30, 50]
EVALUATION_METRICS = ["auc_roc"]
PARAMS_DICT = {
    "random_forest": {
        "n_estimators": [10, 100, 500, 1000],
        "max_depth": [1, 5, 10],
        "min_samples_split": [2, 5, 10],
        "n_jobs": [-1]
    },
    "logistic_regression": {
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"]
    },
    "decision_tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [1, 5, 10],
        "min_samples_split": [2, 5, 50]
    },
    "SVM": {},
    "knn": {
        "n_neighbors": [5, 10, 25, 50],
        "weights": ["uniform", "distance"]
    },
    "ada_boost": {
        "algorithm": ["SAMME.R"],
        "n_estimators": [1, 10, 100, 1000]
    },
    "gradient_boost": {
        "n_estimators": [10, 100],
        "max_depth": [3, 5, 10]
    },
    "bagging": {
        "n_estimators": [10, 100],
        "random_state": [0],
        "n_jobs": [-1]
    }
}
DEFAULT_MODELS = [
    "decision_tree", "random_forest", "knn", "logistic_regression", "bagging"
]
CLASSIFIERS = {
    "random_forest": RandomForestClassifier(),
    "logistic_regression": LogisticRegression(),
    "SVM": svm.SVC(random_state=0, probability=True),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "ada_boost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),
    "gradient_boost": GradientBoostingClassifier(),
    "bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=5))
}
EVALUATION_COLUMNS = [
    "model", "parameters", "test_baseline",
    "train_lbound", "train_ubound", "valid_lbound", "valid_ubound",
    "accuracy_at_1", "precision_at_1", "recall_at_1", "f1_score_at_1",
    "accuracy_at_2", "precision_at_2", "recall_at_2", "f1_score_at_2",
    "accuracy_at_5", "precision_at_5", "recall_at_5", "f1_score_at_5", 
    "accuracy_at_10", "precision_at_10", "recall_at_10", "f1_score_at_10", 
    "accuracy_at_20", "precision_at_20", "recall_at_20", "f1_score_at_20",
    "accuracy_at_30", "precision_at_30", "recall_at_30", "f1_score_at_30",
    "accuracy_at_50", "precision_at_50", "recall_at_50", "f1_score_at_50",
    "auc_roc"
]
SUMMARY_COLUMNS = [
    "model", "precision_at_5", "accuracy_at_5", "f1_score_at_5", 
    "recall_at_5", "auc_roc"
]


def run_pipeline(arguments):

    if not arguments.model:
        models = DEFAULT_MODELS
    else:
        models = [arguments.model]
    for temporal_split in TEMPORAL_SPLITS:
        # Collect train, valid sets and features for this temporal split.
        train_set, valid_set = request_train_valid_sets(*temporal_split)
        features_list = [
            column 
            for column in train_set.columns 
            if column not in RESERVED_COLUMNS + DROP_COLUMN_KEYWORDS
        ]
        # Split train, valid sets into independent, dependent variable sets.
        X_train = train_set[features_list]
        y_train = train_set[TARGET_VARIABLE]
        X_valid = valid_set[features_list]
        y_valid = valid_set[TARGET_VARIABLE]
        data = X_train, y_train, X_valid, y_valid
        # Generate and evaluate models.
        evaluations = generate_models(
            *data, *temporal_split, models, THRESHOLDS
        )
        valid_lb, valid_ub = temporal_split[2:]
        filename = "_".join(["evaluation", "valid", valid_lb, valid_ub]) + ".csv"
        evaluations.to_csv(filename, index=False)
        # Report top performing models on each evaluation metric.
        report_best_models(
            *data, valid_lb, valid_ub, evaluations, EVALUATION_METRICS
        )
        # Report feature importance.
        report_feature_importance(X_train, y_train)


def request_train_valid_sets(train_lb, train_ub, valid_lb, valid_ub):

    '''
    Identify the appropriate train, valid set CSV files by date range
    per standard filenaming convention.

    train_lb (datetime): training set temporal lower bound inclusive.
    train_ub (datetime): training set temporal upper bound exclusive.
    valid_lb (datetime): validation set temporal lower bound inclusive.
    valid_ub (datetime): validation set temporal upper bound exclusive.

    Return train, valid sets (DatFrame, DataFrame)

    '''
    
    # Collect train, valid sets from CSV per standard naming convention.
    train_set_filename = "_".join(["train", train_lb, train_ub]) + ".csv"
    valid_set_filename = "_".join(["valid", valid_lb, valid_ub]) + ".csv"
    train_set = postprocess_data(pd.read_csv(train_set_filename))
    valid_set = postprocess_data(pd.read_csv(valid_set_filename))
    return train_set, valid_set


def postprocess_data(dataframe):
    
    # Get month_issue from earliest_issue column.
    dataframe["month"] = pd.to_datetime(dataframe["earliest_issue"]).dt.month
    # Impute all missingness as zero.
    dataframe = dataframe.fillna(0)
    return dataframe


def generate_models(X_train, y_train, X_valid, y_valid, train_lb, train_ub, \
    valid_lb, valid_ub, models, thresholds):

    '''
    Generate models for a temporal set and evaluate and several thresholds.
    Optimizing for best AUC, track the model with the best performance on
    that metric. Export the final analyses to CSV.

    X_train (DataFrame): features on which to train the model.
    y_train (DataFrame): target variable of training set.
    X_valid (DataFrame): features of which to validate the model.
    y_valid (DataFrame): target variable of the validation set.
    train_lb (datetime): training set temporal lower bound inclusive.
    train_ub (datetime): training set temporal upper bound exclusive.
    valid_lb (datetime): validation set temporal lower bound inclusive.
    valid_ub (datetime): validation set temporal upper bound exclusive.
    models (list): representation of models to run.
    thresholds (list): representation of percentiles at which to evaluate.

    Return evaluation (DataFrame)

    '''

    evaluations = pd.DataFrame(columns=EVALUATION_COLUMNS)
    best_auc = 0.0
    best_model = []
    for model in models:
        print("Running model " + model)
        classifier = CLASSIFIERS[model]
        parameters_to_run = PARAMS_DICT[model]
        # Loop through varying paramets for each model
        for parameters in ParameterGrid(parameters_to_run):
            constants = [
                model, parameters, np.mean(y_valid), 
                train_lb, train_ub, valid_lb, valid_ub
            ]
            classifier.set_params(**parameters)
            classifier.fit(X_train, y_train)
            y_proba = classifier.predict_proba(X_valid)[:, 1]
            # Evaluate model on each threshold.
            metrics = []
            for threshold in thresholds:
                y_predi = [
                    1 if y > np.percentile(y_proba, threshold) else 0
                    for y in y_proba
                ]
                metrics.extend([
                    accuracy_score(y_valid, y_predi),
                    precision_score(y_valid, y_predi),
                    recall_score(y_valid, y_predi),
                    f1_score(y_valid, y_predi)
                ])
            auc_roc = roc_auc_score(y_valid, y_proba)
            # Write to evaluation dataframe.
            evaluations.loc[len(evaluations)] = constants + metrics + [auc_roc]
            if auc_roc > best_auc:
                best_auc = auc_roc
                y_proba = pd.Series(y_proba, name="score")
                best_model = [X_valid, y_valid, y_proba]
    # Write validation set and scores from the best model to CSV.
    filename = "_".join(["results", "valid", valid_lb, valid_ub]) + ".csv"
    pd.concat(best_model, axis=1).to_csv(filename, index=False)
    return evaluations


def report_best_models(X_train, y_train, X_valid, y_valid, valid_lb, valid_ub, \
    evaluations, metrics):

    '''
    Report the best performing models on the metrics given for consideration.
    Print the evaluations to screen and plot its precision-recall and ROC curves.

    X_train (DataFrame): features on which to train the model.
    y_train (DataFrame): target variable of training set.
    X_valid (DataFrame): features of which to validate the model.
    y_valid (DataFrame): target variable of the validation set.
    valid_lb (datetime): validation set temporal lower bound inclusive.
    valid_ub (datetime): validation set temporal upper bound exclusive.
    evaluations (DataFrame): evaluations from generate_models().
    metrics (list): representation of metrics to consider

    '''

    for metric in metrics:
        print("BEST MODEL FOR " + metric.upper())
        best_index = evaluations[metric].idxmax()
        best_series = evaluations.loc[best_index]
        print(best_series[SUMMARY_COLUMNS])
        create_curves(
            best_series["model"], best_series["parameters"], 
            X_train, y_train, X_valid, y_valid, valid_lb
        )


def report_feature_importance(X_train, y_train):

    '''
    Report feature importance with a simple decision tree classifier.
    Print feature importance to screen.
    
    X_train (DataFrame): features on which to train the model.
    y_train (DataFrame): target variable of training set.

    '''

    classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
    classifier.fit(X_train, y_train)
    feataure_importance = classifier.feature_importances_
    for i in range(len(feataure_importance)):
        if feataure_importance[i] > 0.01:
            print(feataure_importance[i], X_train.columns[i])
    

def create_curves(model, params, X_train, y_train, X_valid, y_valid, valid_lb, \
    threshold=5):
    '''
    Prints area under the curve and creates and saves an ROC and
    precision-recall curves image.

    model: name of machine learning classifer
    params: params for classifier to run
    x_train: pandas dataframe with only features columns of training data
    X_valid: pandas dataframe with only features columns of testing data
    y_train: pandas series with outcome column of training data
    y_valid: pandas series with outcome column of testing data
    valid_lb (datetime): validation set temporal lower bound inclusive.

    '''
    classifier = CLASSIFIERS[model]
    try:
        classifier.set_params(**params)
    except:
        classifier.set_params(**ast.literal_eval(params))
    classifier.fit(X_train, y_train)
    y_proba = classifier.predict_proba(X_valid)[:, 1]
    y_predi = [
        1 if y > np.percentile(y_proba, threshold) else 0
        for y in y_proba
    ]
    auc_roc = roc_auc_score(y_valid, y_proba)
    print("AUC: %.3f" % auc_roc)
    fpr, tpr, _ = roc_curve(y_valid, y_proba)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    roc_title = "ROC " + model + " with " + str(params) + valid_lb
    plt.savefig(roc_title + ".png")
    plt.clf()
    plot_precision_recall_n(model, params, y_valid, y_predi, valid_lb)


# The code below comes from Rayid Ghani's magic loop, found here:
# https://github.com/rayidghani/magicloops

def plot_precision_recall_n(model, params, y_true, y_score, valid_lb):
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
    ax1.plot(pct_above_per_thresh, precision_curve, "b")
    ax1.set_xlabel("percent of population")
    ax1.set_ylabel("precision", color="b")
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, "r")
    ax2.set_ylabel("recall", color="r")
    ax1.set_ylim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])
    p_r_title = "Precision-Recall " + model + " with " + str(params) + valid_lb
    plt.savefig(p_r_title + ".png")
    plt.clf()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_roc = roc_auc_score(y_true, y_score)
    print(auc_roc)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    roc_title = "ROC " + model + " with " + str(params) + valid_lb
    plt.title("ROC")
    plt.savefig(roc_title + ".png")
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the Chicago Entrepreneurship database."
    )
    parser.add_argument(
        "--model",
        help="Models for the pipeline to run.",
        dest="model",
        default=None
    )
    arguments = parser.parse_args()
    run_pipeline(arguments)

