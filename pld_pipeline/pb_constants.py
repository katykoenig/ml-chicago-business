'''
Plumbum: a machine learning pipeline.
Constants module

Patrick Lavallee Delgado
University of Chicago, CS & Harris MSCAPP '20
May 2019

'''

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


_DEFAULT_TEMPORAL_SELECT = '''
SELECT * 
FROM $db_name 
WHERE DATETIME($temporal_variable) >= DATETIME('$l_bound') 
AND DATETIME($temporal_variable) < DATETIME('$u_bound');
'''
_BASE_ESTIMATOR = [
    LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        penalty="l2",
        random_state=0,
        solver="lbfgs"
    ),
    DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_split=2,
        random_state=0
    )
]
_C_PENALTY = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
_MAX_DEPTH = [1, 3, 5, 10, 20, 50, 100]
_MAX_FEATURES = ["sqrt", "log2", None]
_MIN_SAMPLES_SPLIT = [2, 5, 10]
_N_ESTIMATORS = [1, 10, 100, 1000, 10000]
_N_JOBS = [-1]
_RANDOM_STATE = [0]
_CRITERION = ["gini", "entropy"]
_DEFAULT_METHODS = {
    "adaboost": (
        AdaBoostClassifier(),
        {
            "algorithm": ["SAMME", "SAMME.R"],
            "base_estimator": _BASE_ESTIMATOR,
            "n_estimators": _N_ESTIMATORS,
            "random_state": _RANDOM_STATE
        }
    ),
    "bagging": (
        BaggingClassifier(),
        {
            "base_estimator": _BASE_ESTIMATOR,
            "n_estimators": _N_ESTIMATORS,
            "n_jobs": _N_JOBS,
            "random_state": _RANDOM_STATE
        }
    ),
    "bayes": (
        GaussianNB(),
        {}
    ),
    "extra_trees": (
        ExtraTreesClassifier(),
        {
            "criterion": _CRITERION,
            "max_depth": _MAX_DEPTH,
            "max_features": _MAX_FEATURES,
            "min_samples_split": _MIN_SAMPLES_SPLIT,
            "n_estimators": _N_ESTIMATORS,
            "n_jobs": _N_JOBS,
            "random_state": _RANDOM_STATE
        }
    ),
    "gausian_mixture": (
        GaussianMixture(),
        {
            # add any parameters here.
        }
    ),
    "gradient": (
        GradientBoostingClassifier(),
        {
            "learning_rate" : [0.001, 0.01, 0.05, 0.1, 0.5],
            "n_estimators": _N_ESTIMATORS,
            "max_depth": _MAX_DEPTH,
            "random_state": _RANDOM_STATE,
            "subsample" : [0.1, 0.5, 1.0]
        }
    ),
    "knn": (
        KNeighborsClassifier(),
        {
            "algorithm": ["auto","ball_tree","kd_tree"],
            "n_neighbors": [1, 5, 10, 25, 50, 100],
            "weights": ["uniform","distance"]
        }
    ),
    "logistic": (
        LogisticRegression(),
        {        
            "C": _C_PENALTY,
            "max_iter": [1000],
            "multi_class": ["multinomial"],
            "penalty": ["l2"],
            "random_state": _RANDOM_STATE,
            "solver": ["lbfgs"]
        }
    ),
    "random_forest": (
        RandomForestClassifier(),
        {
            "criterion": _CRITERION,
            "max_depth": _MAX_DEPTH,
            "max_features": _MAX_FEATURES,
            "min_samples_split": _MIN_SAMPLES_SPLIT,
            "n_estimators": _N_ESTIMATORS,
            "n_jobs": _N_JOBS,
            "random_state": _RANDOM_STATE
        }
    ),
    "svm": (
        SVC(),
        {
            "C": _C_PENALTY,
            "gamma": ["auto"],
            "kernel": ["linear"],
            "probability": [True],
            "random_state": _RANDOM_STATE
        }
    ),
    "tree": (
        DecisionTreeClassifier(),
        {
            "criterion": _CRITERION,
            "max_depth": _MAX_DEPTH,
            "min_samples_split": _MIN_SAMPLES_SPLIT,
            "random_state": _RANDOM_STATE
        }
    )
}
_UNSUPERVISED_LEARNING_METHODS = ["gausian_mixture"]
_DEFAULT_THRESHOLDS = [1, 2, 5, 10, 20, 30, 100]
_DEFAULT_MAX_DISCRETIZATION = 100
_DEFAULT_OPTIMIZING_METRICS = ["accuracy", "precision", "recall", "auc"]
_DEFAULT_VALID_SIZE = 0.3
_DEFAULT_RANDOM_STATE = _RANDOM_STATE[0]
_DEFAULT_EVALUATION_COLUMNS = ["method", "t_index", "train_l_bound", 
    "train_u_bound", "valid_l_bound", "valid_u_bound", "shape", "features", 
    "p_index", "parameters", "computation_time", "auc", "threshold",
    "accuracy", "precision", "recall", "f1"]
select_methods = {
    "adaboost": (
        AdaBoostClassifier(),
        {
            "algorithm": ["SAMME", "SAMME.R"],
            "base_estimator": _BASE_ESTIMATOR,
            "n_estimators": _N_ESTIMATORS,
            "random_state": _RANDOM_STATE
        }
    ),
    "bagging": (
        BaggingClassifier(),
        {
            "base_estimator": _BASE_ESTIMATOR,
            "n_estimators": _N_ESTIMATORS,
            "n_jobs": _N_JOBS,
            "random_state": _RANDOM_STATE
        }
    ),
    "bayes": (
        GaussianNB(),
        {}
    ),
    "logistic": (
        LogisticRegression(),
        {        
            "C": _C_PENALTY,
            "max_iter": [1000],
            "multi_class": ["multinomial"],
            "penalty": ["l2"],
            "random_state": _RANDOM_STATE,
            "solver": ["lbfgs"]
        }
    ),
    "random_forest": (
        RandomForestClassifier(),
        {
            "criterion": ["entropy"],
            "max_depth": [5, 10, 50],
            "max_features": _MAX_FEATURES,
            "min_samples_split": _MIN_SAMPLES_SPLIT,
            "n_estimators": _N_ESTIMATORS,
            "n_jobs": _N_JOBS,
            "random_state": _RANDOM_STATE
        }
    ),
    "tree": (
        DecisionTreeClassifier(),
        {
            "criterion": ["entropy"],
            "max_depth": [5, 10, 50],
            "min_samples_split": _MIN_SAMPLES_SPLIT,
            "random_state": _RANDOM_STATE
        }
    )
}
short_methods = {
    "tree": (
        DecisionTreeClassifier(),
        {
            "criterion": ["entropy"],
            "max_depth": [10],
            "min_samples_split": [2],
            "random_state": [0]
        }
    )
}