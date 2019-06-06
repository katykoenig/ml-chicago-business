'''
Promoting Sustained Entrepreneurship in Chicago
Machine Learning for Public Policy
University of Chicago, CS & Harris School of Public Policy
June 2019

Rayid Ghani (@rayidghani)
Katy Koenig (@katykoenig)
Eric Langowski (@erhla)
Patrick Lavallee Delgado (@lavalleedelgado)

Project machine learning pipeline built on:

Plumbum: a machine learning pipeline.
Patrick Lavallee Delgado
University of Chicago, CS & Harris MSCAPP '20
June 2019

'''

import os
import csv
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, ParameterGrid
import sqlite3
import chicago_entrepreneurship_pb_constants as pbk
import chicago_entrepreneurship_pb_plotting as pbp


class Plumbum:

    def __init__(self, name="Plumbum"):

        '''
        Initialize a Plumbum object with default attributes. Plumbum connects
        to the database in the present working directory that shares its name,
        or builds that database from the CSV file at the given path.

        name (str): representation of Plumbum object and database name.
        path_to_csv (str): location at which to find CSV file of the data.

        Return Plumbum object (Plumbum).

        '''

        self._reset_modeling_defaults()
        self._db_name = "_".join(name.lower().split())
        path_convention = os.path.dirname(__file__) + "/" + self._db_name
        self._db_path =  path_convention + "_db.sqlite3"
        self._ev_path = path_convention + "_models.csv"
        self._target_variable = None
        self._temporal_splits = []
        self._temporal_interval = np.timedelta64(2, "Y")
        self._best_models = {}
    

    def _db_open(self):

        '''
        Open connection to the database.

        '''

        self._db_con = sqlite3.connect(self._db_path)
        self._db_cur = self._db_con.cursor()

    
    def _db_close(self):

        '''
        Close connection to the database.

        '''

        self._db_con.close()

    
    def _reset_modeling_defaults(self):

        self._methods = pbk._DEFAULT_METHODS
        self._thresholds = pbk._DEFAULT_THRESHOLDS
        self._max_discretization = pbk._DEFAULT_MAX_DISCRETIZATION
        self._optimizing_metrics = pbk._DEFAULT_OPTIMIZING_METRICS


    def _target_variable_exists(self):

        return self._target_variable is not None


    @property
    def target_variable(self):

        '''
        Verify the variable on which to generate classification models.

        Return target variable (str).

        '''

        return self._target_variable
    

    @target_variable.setter
    def target_variable(self, variable):

        '''
        Set the target variable.

        variable (str): representation of existing column in the database.

        '''

        self._target_variable = variable
        
    
    @property
    def temporal_splits(self):

        '''
        Verify the temporal splits on which to control temporal modeling.

        Return temporal splits (list).

        '''

        return self._temporal_splits


    @temporal_splits.setter
    def temporal_splits(self, temporal_splits):

        '''
        Set the temporal splits.

        temporal_splits (list of tuples): lower and upper bounds for training
        set and lower and upper bounds for validation set. [lower, upper)

        '''

        self._temporal_splits = temporal_splits


    @property
    def temporal_interval(self):

        '''
        Verify the temporal interval on which to control temporal modeling.

        Return temporal interval (np.timedelta64).

        '''

        return self._temporal_interval


    @temporal_interval.setter
    def temporal_interval(self, temporal_interval):

        '''
        Set the temporal interval.

        temporal_interval (np.timedelta64): period between prediction date and
        outome date.

        '''

        self._temporal_interval = temporal_interval
        
        
    @property
    def methods(self):

        '''
        Verify the classifiers and parameters with which to generate models.
        Provide multiple values for classifier parameters in a list to generate
        models on permutations of parameterizations of that classifer.

        Return methods (dict).

        '''

        return self._methods


    @methods.setter
    def methods(self, methods):

        '''
        Set the classifiers and their parameters.

        methods (dict): mapping of classifier names to classifier objects and
        sub-mapping of parameter names to parameter values.

        Example:
            {
                "tree": (
                    DecisionTreeClassifier(),
                    {
                        max_depth: [1, 5, 10, 100],
                        random_state: [0]
                    }
                )
            }

        '''

        self._methods = methods


    @property
    def thresholds(self):

        '''
        Verify the percentiles at which to identify the minimum scores for
        classification to the positive class.

        Return thresholds (list).

        '''
        
        return self._thresholds


    @thresholds.setter
    def thresholds(self, thresholds):

        '''
        Set the thresholds.
        
        thresholds (list): collection of integers in the range [0, 100].

        '''

        assert isinstance(thresholds, list)
        self._thresholds = thresholds


    @property
    def max_discretization(self):

        '''
        Verify the length of a domain by which to distinguish a variable as 
        discrete or continuous. This ceiling averts unintentional or unweildly
        binarization of variables.

        Return maximum discretization (int).

        '''

        return self._max_discretization


    @max_discretization.setter
    def max_discretization(self, max_discretization):

        '''
        Set the maximum discretization.

        max_discretization (int): maximum allowable length of the domain of a
        discrete or categorical variable.

        '''

        assert isinstance(max_discretization, int)
        self._max_discretization = max_discretization

    
    @property
    def optimizing_metrics(self):

        '''
        Verify the metrics with which to optimize for the best classifier.

        Return optimizing metrics (list).

        '''

        return self._optimizing_metrics

    
    @optimizing_metrics.setter
    def optimizing_metrics(self, metrics):

        '''
        Set the optimizing metrics.

        '''

        assert isinstance(metrics, list)
        self._optimizing_metrics = metrics


    def classify(self, methods=None, o_thold=5):

        '''
        Classify the data on the target variable in two major operations:
        (1) impute missingness and generate features per training data of each
        temporal split, (2) generate and evaluate models per temporal split,
        parameterization.

        methods (list): representation of methods by which to model.
        o_thold (int): threshold at which to identify optimal models.

        Return the best models (set of sklearn classifier objects).

        '''

        # Validate optimization threshold.
        assert o_thold in self._thresholds, \
            "Expected percentile from set of thresholds."
        # Generate models with temporal validation. Note that this specific 
        # implementation does not support out-of-sample validation.
        for t_index, temporal_split in enumerate(self._temporal_splits):
            # Request and generate features on the training data.
            train_set, train_set_feature_map = self._generate_features(
                data=self._db_request(*temporal_split, train=True)
            )
            # Request and generate features on the validation data per the
            # approach learned on the training data.
            valid_set, _ = self._generate_features(
                data=self._db_request(*temporal_split, train=False),
                feature_map=train_set_feature_map
            )
            # Split data and validate methods for either supervised or
            # unsupervised learning.
            X_train, y_train, X_valid, y_valid, methods = \
                self._prepare_learning_curriculum(
                    train_set, valid_set, methods
                )
            # Generate models per this machine learning curriculum.
            self._generate_models(
                X_train, y_train, X_valid, y_valid, methods, t_index,
                o_thold
            )
        # Plot curves of the best models and return their classifiers.
        # return self._unpack_best_models()


    def _db_request(self, tl_bound, tu_bound, vl_bound, vu_bound, train):

        '''
        Deploy temporal select statement to request data from the database.

        tl_bound (Timestamp): temporal lower bound inclusive for training set.
        tu_bound (Timestamp): temporal upper bound exclusive for training set.
        vl_bound (Timestamp): temporal lower bound inclusive for validation set.
        vu_bound (Timestamp): temporal upper bound exclusive for validation set.
        train (bool): whether to request a training or validation set.

        Return data (DataFrame).

        '''

        # Set interval and join limit constants.
        interval = self._temporal_interval
        JOIN_LIMIT = 60
        # Convert np.datetime64 to pd.Timestamp for convenience.
        tl_bound = pd.Timestamp(tl_bound)
        tu_bound = pd.Timestamp(tu_bound)
        vl_bound = pd.Timestamp(vl_bound)
        vu_bound = pd.Timestamp(vu_bound)
        # Toggle between bounds for training or validation set.
        if train:
            l_bound, u_bound = tl_bound, tu_bound
        else:
            l_bound, u_bound = vl_bound, vu_bound
        # Open connection.
        self._db_open()
        # Request and collect storefronts in this period.
        select_storefronts = f'''
        WITH 
            storefronts_general AS ( 
                SELECT 
                    account_number || '-' || site_number AS sf_id, 
                    MIN(issue_date) AS earliest_issue, 
                    MAX(expiry_date) AS latest_issue 
                FROM licenses 
                WHERE DATETIME(expiry_date) >= DATETIME('{l_bound}') 
                AND DATETIME(issue_date) < DATETIME('{u_bound}') 
                GROUP BY account_number, site_number 
            ), 
            storefronts_general_future AS (
                SELECT 
                    account_number || '-' || site_number AS sf_id 
                FROM licenses 
                WHERE DATETIME(expiry_date) >= DATETIME('{l_bound + interval}') 
                AND DATETIME(issue_date) < DATETIME('{u_bound + interval}') 
                GROUP BY account_number, site_number 
            ), 
            storefronts_success AS ( 
                SELECT 
                    sf_id, 
                    earliest_issue, 
                    latest_issue, 
                    CASE WHEN 
                        sf_id IN (SELECT sf_id FROM storefronts_general_future) 
                        THEN 1 
                        ELSE 0 
                    END successful 
                FROM storefronts_general 
            ), 
            storefronts_location AS ( 
                SELECT 
                    account_number || '-' || site_number AS sf_id, 
                    block, 
                    RANK() OVER ( 
                        PARTITION BY account_number, site_number 
                        ORDER BY issue_date DESC 
                    ) AS last_location 
                FROM licenses 
                WHERE block IS NOT NULL 
                AND DATETIME(expiry_date) >= DATETIME('{l_bound}') 
                AND DATETIME(issue_date) < DATETIME('{u_bound}') 
            ), 
            storefronts_blocks AS ( 
                SELECT 
                    COUNT( 
                        DISTINCT account_number || '-' || site_number 
                    ) AS storefronts_on_block, 
                    block 
                FROM licenses 
                WHERE block IS NOT NULL 
                AND DATETIME(expiry_date) >= DATETIME('{l_bound}') 
                AND DATETIME(issue_date) < DATETIME('{u_bound}') 
                GROUP BY block 
            )
        SELECT DISTINCT 
            sf_id, 
            earliest_issue, 
            latest_issue, 
            storefronts_on_block, 
            block, 
            successful 
        FROM storefronts_success 
        JOIN storefronts_location USING (sf_id) 
        JOIN storefronts_blocks USING (block) 
        WHERE last_location = 1;
        '''
        storefronts = pd.read_sql(select_storefronts, self._db_con)
        # Request and collect licenses extant in training period by storefront.
        select_storefronts_licenses = f'''
        SELECT * 
        FROM ( 
            SELECT account_number || '-' || site_number AS sf_id 
            FROM licenses 
            WHERE DATETIME(expiry_date) >= DATETIME('{l_bound}') 
            AND DATETIME(issue_date) < DATETIME('{u_bound}') 
            GROUP BY account_number, site_number 
        ) AS storefronts 
        '''
        select_extant_licenses = self._db_cur.execute(f'''
            SELECT DISTINCT license_code 
            FROM licenses 
            WHERE DATETIME(expiry_date) >= DATETIME('{tl_bound}') 
            AND DATETIME(issue_date) < DATETIME('{tu_bound}') 
            '''
        ) 
        extant_licenses = select_extant_licenses.fetchall()
        # Request licenses individually and join relations in batches.
        licenses_join_complete = []
        licenses_join_queue = []
        for i, extant_license in enumerate(extant_licenses):
            license_lable = "_".join(extant_license[0].lower().split())
            licenses_join_queue.append(f'''
                LEFT JOIN (
                    SELECT 
                        account_number || '-' || site_number AS sf_id, 
                        COUNT(license_code) AS license_{license_lable} 
                    FROM licenses 
                    WHERE license_code = '{extant_license[0]}' 
                    AND DATETIME(expiry_date) >= DATETIME('{l_bound}') 
                    AND DATETIME(issue_date) < DATETIME('{u_bound}') 
                    GROUP BY account_number, site_number 
                ) AS L_{license_lable} USING (sf_id) 
                '''
            ) 
            # Execute these joins at the join limit or with the last relation.
            if i % JOIN_LIMIT == 0 or i == len(extant_licenses) - 1:
                batch = pd.read_sql(
                    select_storefronts_licenses + " ".join(licenses_join_queue),
                    self._db_con
                )
                licenses_join_complete.append(batch.fillna(0))
                licenses_join_queue = []
        # Merge batches into one dataframe.
        licenses = licenses_join_complete.pop()
        for batch in licenses_join_complete:
            licenses = licenses.merge(batch, on="sf_id")
        # Request and collect crimes extant in training period by block.
        select_crime_general = f'''
        WITH 
            domestic AS ( 
                SELECT block, COUNT(domestic) AS domestic_sum 
                FROM crimes 
                WHERE DATETIME(date) >= DATETIME('{tl_bound}') 
                AND DATETIME(date) < DATETIME('{tu_bound}') 
                AND domestic = 'True' 
                GROUP BY block 
            ), 
            arrest AS ( 
                SELECT block, COUNT(arrest) AS arrest_sum 
                FROM crimes 
                WHERE DATETIME(date) >= DATETIME('{tl_bound}') 
                AND DATETIME(date) < DATETIME('{tu_bound}') 
                AND arrest = 'True' 
                GROUP BY block 
            ),
            sum AS (
                SELECT block, COUNT(crime) AS crime_sum 
                FROM crimes 
                WHERE DATETIME(date) >= DATETIME('{tl_bound}') 
                AND DATETIME(date) < DATETIME('{tu_bound}') 
                GROUP BY block 
            ) 
        SELECT DISTINCT block 
        FROM blocks 
        LEFT JOIN domestic USING (block) 
        LEFT JOIN arrest USING (block) 
        LEFT JOIN sum USING (block);
        '''
        crime_general = pd.read_sql(select_crime_general, self._db_con)
        select_crimes_blocks = f'''
        SELECT * 
        FROM (
            SELECT DISTINCT block 
            FROM blocks
        ) AS blocks 
        '''
        select_extant_crimes = self._db_cur.execute(f'''
            SELECT DISTINCT crime 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{tl_bound}') 
            AND DATETIME(date) < DATETIME('{tu_bound}');
            '''
        )
        extant_crimes = select_extant_crimes.fetchall()
        # Request crimes individually and join relations in batches.
        crimes_join_complete = [crime_general]
        crimes_join_queue = []
        for i, extant_crime in enumerate(extant_crimes):
            crime_label = "_".join(
                extant_crime[0].lower() \
                .replace("(", "").replace(")", "").replace("-", "") \
                .split()
            )
            crimes_join_queue.append(f'''
                LEFT JOIN (
                    SELECT block, 
                    COUNT(crime) AS crime_{crime_label} 
                FROM crimes 
                WHERE crime = '{extant_crime[0]}' 
                AND DATETIME(date) >= DATETIME('{l_bound}') 
                AND DATETIME(date) < DATETIME('{u_bound}') 
                GROUP BY block 
                ) AS C_{crime_label} USING (block) 
                '''
            )
            # Execute these joins at the join limit or with the last relation.
            if i % JOIN_LIMIT == 0 or i == len(extant_crimes) - 1:
                batch = pd.read_sql(
                    select_crimes_blocks + " ".join(crimes_join_queue),
                    self._db_con
                )
                crimes_join_complete.append(batch.fillna(0))
                crimes_join_queue = []
        # Merge batches into one dataframe.
        crimes = crimes_join_complete.pop()
        for batch in crimes_join_complete:
            crimes = crimes.merge(batch, on="block")
        # Request census data.
        census = pd.read_sql("SELECT * FROM census", self._db_con)
        # Close connection.
        self._db_close()
        # Join storefronts, licenses, crimes, census data.
        data = storefronts.merge(licenses, on="sf_id").merge(crimes, on="block")
        data["block_group"] = data["block"].apply(lambda x: x // 1000)
        data = data.merge(census, on="block_group")
        # Compose the filename for this temporal set.
        if train:
            set_type = "train"
        else:
            set_type = "valid"
        filename = "_".join(
            [set_type, str(l_bound.date()), str(u_bound.date())]
        )
        # Save the dataframe to CSV and return the dataframe for modeling.
        data.to_csv(filename + ".csv", index=False)
        return data


    def _generate_features(self, data, feature_map={}):

        '''
        Generate features from variables in the data by scaling or discretizing
        continuous variables and binarizing discrete variables. Specify
        feature_map to apply the feature generation done on the training data
        to the validation data.

        data (DataFrame): data from which to generate features.
        feature_map (dict): mapping of feature generation on training data.

        Return featurized data (DataFrame), bin map (dict).

        '''

        data, feature_map = self._drop_uninsightful(data, feature_map)
        data, feature_map = self._min_max_scale_continuous(data, feature_map)
        data, feature_map = self._discretize_continuous(data, feature_map)
        data, feature_map = self._binarize_discrete(data, feature_map)
        return data, feature_map


    def _drop_uninsightful(self, data, feature_map):

        '''
        Drop variables least likely to offer meaningful insight in a model.
        These include string type variables whose unique values exceed the
        maximum discretization, any variable with zero variance, and any
        variable whose unique values equal the number of records in the data.
        Specify feature_map to drop the same variables from training data with
        validation data.

        data (DataFrame): data to justify.
        feature_map (dict): mapping of feature generation on training data.

        Return insightful data (DataFrame), feature map (dict).

        '''

        # Identify uninsightful variables to drop.
        if "uninsightful" not in feature_map:
            feature_map["uninsightful"] = [
                variable
                for variable in data.columns
                if self._validate_for_uninsightful(data[variable])
            ]
        # Drop uninsightful variables.
        return data.drop(columns=feature_map["uninsightful"]), feature_map


    def _validate_for_uninsightful(self, variable):

        return (pd.api.types.is_string_dtype(variable) and \
            variable.nunique() > self._max_discretization) or \
            (pd.api.types.is_numeric_dtype(variable) and \
            variable.var() == 0.0) or \
            variable.nunique() == variable.shape[0] and \
            variable.name != self._target_variable
        

    def _min_max_scale_continuous(self, data, feature_map):

        '''
        Min-max scale continuous variables that would not discretize into bins
        numbering in the range of the specified maximum discretization. Scale
        these variables such that they assume values in the range [0, 1.
        Specify feature_map to apply the scaling done on the training data to
        the validation data.

        data (DataFrame): data to scale.
        feature_map (dict): mapping of feature generation on training data.

        Return scaled data (DataFrame), feature map (dict).
        
        '''

        # Identify continuous variables to scale if there is no feature map.
        if "scale" not in feature_map:
            feature_map["scale"] = {
                variable: None
                for variable in data.columns
                if self._validate_for_scaling(data[variable])
            }
        scaling_map = feature_map["scale"]
        # Impute missingness in continuous variables to scale.
        data = self._impute_continuous(data, scaling_map.keys())
        for variable in scaling_map:
            # Create scaling function for variable if there is no feature map.
            if scaling_map.get(variable) is None:
                variable_min = data[variable].min()
                variable_max = data[variable].max()
                def min_max_scaling_function(x):
                    return (x - variable_min) / (variable_max - variable_min)
                scaling_map[variable] = min_max_scaling_function
            # Apply scaling function to the variable.
            data[variable] = data[variable].apply(scaling_map[variable])
        # Save scaling functions to the feature map.
        feature_map["scale"].update(scaling_map)
        return data, feature_map


    def _validate_for_scaling(self, variable):

        return variable.nunique() > self._max_discretization and \
            (variable.shape[0] // variable.std() == 0 or \
            variable.shape[0] // variable.std() > self._max_discretization) \
            and not (variable.min() == 0.0 and variable.max() == 1.0) and \
            variable.name != self._target_variable
        

    def _impute_continuous(self, data, continuous_variables):

        '''
        Impute missingness in continuous variables with the variable mean.

        data (DataFrame): data to impute.
        continuous_variables (list): representation of continuous variables.

        Return imputed data (DataFrame).
        
        '''

        for variable in continuous_variables:
            data[variable].replace(
                to_replace=["", np.nan],
                value=data[variable].mean(),
                inplace=True
            )
        return data


    def _discretize_continuous(self, data, feature_map):

        '''
        Discretize continuous variables that would discretize into bins
        numbering in the range of the specified maximum discretization.
        Discretize these variables into quantiles numbering the quotient of the
        length and standard deviation of the variable. Specify feature_map to
        apply the discretization done on the training data to the validation
        data.

        data (DataFrame): data to discretize.
        feature_map (dict): mapping of feature generation on training data.

        Return discretized data (DataFrame), feature map (dict).
        
        '''

        # Identify continuous variables to discretize if there is no map.
        if "discretize" not in feature_map:
            feature_map["discretize"] = {
                variable: None
                for variable in data.columns
                if self._validate_for_discretization(data[variable])
            }
        discretization_map = feature_map["discretize"]
        # Impute missingness in continuous variables to discretize.
        data = self._impute_continuous(data, discretization_map.keys())
        for variable in discretization_map:
            # Discretize variable by quantiles if there is no feature map.
            if discretization_map.get(variable) is None:
                data[variable], bins = pd.qcut(
                    x=data[variable],
                    q=int(data.shape[0] // data[variable].std()),
                    retbins=True,
                    duplicates="drop"
                )
                discretization_map[variable] = bins
            # Discretize variable per feature map otherwise.
            else:
                data[variable] = pd.cut(
                    x=data[variable],
                    bins=discretization_map[variable]
                )
        # Save discretization bins to the feature map.
        feature_map["discretize"].update(discretization_map)
        return data, feature_map


    def _validate_for_discretization(self, variable):

        return variable.nunique() > self._max_discretization and \
            variable.shape[0] // variable.std() != 0 and \
            variable.shape[0] // variable.std() <= self._max_discretization \
            and not (variable.min() == 0.0 and variable.max() == 1.0) and \
            variable.name != self._target_variable


    def _impute_discrete(self, data, discrete_variables):

        '''
        Impute missingness in discrete variables with the variable mode.

        data (DataFrame): data to impute.
        discrete_variables (list): representation of discrete variables.

        Return imputed data (DataFrame).

        '''

        for variable in discrete_variables:
            data[variable].replace(
                to_replace=["", np.nan],
                value=data[variable].value_counts().idxmax(),
                inplace=True
            )
        return data


    def _binarize_discrete(self, data, feature_map):

        '''
        Binarize discrete variables into indicators for each unique value in
        the variable. Specify feature_map to apply the binarization done on the
        training data to the validation data.

        data (DataFrame): data to binarize.
        feature_map (dict): mapping of feature generation on training data.

        Return binarized data (DataFrame), feature map (dict).
        
        '''

        # Identify discrete variables to binarize if there is no feature map.
        if "binarize" not in feature_map:
            feature_map["binarize"] = {
                variable: None
                for variable in data.columns
                if self._validate_for_binarization(data[variable])
            }
        binarization_map = feature_map["binarize"]
        # Impute missingness in discrete variables to binarize.
        data = self._impute_discrete(data, binarization_map.keys())
        for variable in binarization_map:
            # Identify unique values if there is no feature map.
            if binarization_map.get(variable) is None:
                binarization_map[variable] = data[variable].unique()
            # Binarize variable per unique values.
            for value in binarization_map[variable]:
                value_label = "_".join(str(value).lower().split())
                binarized_label = str(variable) + "_is_" + value_label
                data[binarized_label] = 0
                data.loc[data[variable] == value, binarized_label] = 1
            data.drop(columns=variable, inplace=True)
        # Save binarization values to the feature map.
        feature_map["binarize"].update(binarization_map)
        return data, feature_map


    def _validate_for_binarization(self, variable):

        return not (variable.min() == 0.0 and variable.max() == 1.0)


    def _prepare_learning_curriculum(self, train_set, valid_set, methods):

        '''
        Prepare training data, validation data, and methods for supervised or
        unsupervised learning per whether the target variable exists.

        train_set (DataFrame): feature-generated training data.
        valid_set (DataFrame): feature-generated validation data.
        methods (list): representation of methods by which to model.

        Return training X and y, validation X and y, methods (dict).

        '''

        if not methods:
            methods = self._methods
        # Pursue supervised learning if there is a target variable.
        if self._target_variable_exists():
            X_train = train_set.drop(columns=self._target_variable)
            y_train = train_set[self._target_variable]
            X_valid = valid_set.drop(columns=self._target_variable)
            y_valid = valid_set[self._target_variable]
            # Ensure specified methods exist for supervised learning.
            valid_methods = {
                method: self._methods[method]
                for method in methods
                if method in self._methods.keys() and \
                    method not in pbk._UNSUPERVISED_LEARNING_METHODS
            }
        # Pursue unsupervised learning otherwise.
        else:
            X_train = train_set
            y_train = None
            X_valid = valid_set
            y_valid = None
            # Ensure specified methods exist for unsupervised learning.
            valid_methods = {
                method: self._methods[method]
                for method in methods
                if method in self._methods.keys() and \
                    method in pbk._UNSUPERVISED_LEARNING_METHODS
            }
        return X_train, y_train, X_valid, y_valid, valid_methods


    def _generate_models(self, X_train, y_train, X_valid, y_valid, methods, \
        t_index, o_thold):

        '''
        Generate models of a temporal set per specified methods and available
        parameterizations.

        X_train (DataFrame): features on which to train the model.
        y_train (DataFrame): target variable of training set.
        X_valid (DataFrame): features of which to validate the model.
        y_valid (DataFrame): target variable of the validation set.
        methods (dict): mapping of classifiers and parameters for modeling.
        t_index (int): temporal set index.
        o_thold (int): threshold at which to identify optimal models.

        '''

        # Identify the next classification method by which to model.
        for method, methodology in methods.items():
            classifier, parameters = methodology
            parameterizations = ParameterGrid(parameters)
            # Identify the next parameterization for the classifier.
            for p_index, parameterization in enumerate(parameterizations):
                modeling_time = time.time()
                classifier = classifier.set_params(**parameterization)
                classifier = classifier.fit(X_train, y_train)
                y_proba = classifier.predict_proba(X_valid)[:, 1]
                model_k=self._collate_model_constants(
                    method=method,
                    t_index=t_index,
                    shape=X_train.shape,
                    features=list(X_train.columns),
                    p_index=p_index,
                    params=classifier.get_params(),
                    time=time.time() - modeling_time
                )
                # Evaluate the model on performance metrics.
                self._evaluate_model(
                    classifier, y_valid, y_proba, model_k, o_thold
                )


    def _collate_model_constants(self, method, t_index, shape, features, \
        p_index, params, time):

        '''
        Collate model constants for inclusion with evaluations at thresholds.
        Omit temporal split bounds if the temporal variable does not exist.

        method (str): representation of modeling method.
        t_index (int): temporal set index.
        shape (int): temporal set shape in terms of observations, features.
        features (list): representation of features in model.
        p_index (int): parameterization index.
        params (dict): representation of classifier parameterization.
        time (float): modeling computation time.

        Return model constants (dict).

        '''

        model_constants = {
            "method": method,
            "t_index": t_index,
            "shape": shape,
            "features": features,
            "p_index": p_index,
            "parameters": params,
            "computation_time": time,
            "train_l_bound": self._temporal_splits[t_index][0],
            "train_u_bound": self._temporal_splits[t_index][1],
            "valid_l_bound": self._temporal_splits[t_index][2],
            "valid_u_bound": self._temporal_splits[t_index][3]
            }
        return model_constants


    def _evaluate_model(self, classifier, y_valid, y_proba, model_k, o_thold):

        '''
        Evaluate model at each threshold. Calculate corresponding precision,
        recall, f1 scores.

        classifier (sklearn object): classification engine for this model.
        y_valid (array): validation set of target variable.
        y_proba (array): classification probability set of target variable.
        model_k (dict): mapping of model constants.
        o_thold (int): threshold at which to identify optimal models.

        '''

        evaluations = []
        # Identify the next threshold at which to evaluate the model.
        for threshold in self._thresholds:
            # Assign scores above threshold percentile to the positive class.
            y_predi = [
                1 if y > np.percentile(y_proba, threshold) else 0
                for y in y_proba
            ]
            # Calculate and collate metrics at this threshold.
            if sum(y_predi) == 0:
                precision = 0.0
                f1 = 0.0
            else:
                precision = metrics.precision_score(y_valid, y_predi)
                f1 = metrics.f1_score(y_valid, y_predi)
            evaluation = {
                "auc": metrics.roc_auc_score(y_valid, y_proba),
                "threshold": threshold,
                "accuracy": metrics.accuracy_score(y_valid, y_predi),
                "precision": precision,
                "recall": metrics.recall_score(y_valid, y_predi),
                "f1": f1
            }
            evaluation.update(model_k)
            evaluations.append(evaluation)
            # Check whether this model outperforms others on any metric.
            if threshold == o_thold:
                self._optimize_metrics(
                    classifier, evaluation, y_valid, y_proba
                )
        # Write all evaluations of this model to CSV.
        self._write_evaluations(evaluations)


    def _optimize_metrics(self, classifier, evaluation, y_valid, y_proba):

        '''

        classifier (sklearn object): classification engine for this model.
        evaluation (dict): mapping of model metrics and attributes.
        y_valid (array): validation set of target variable.
        y_proba (array): classification probability set of target variable.

        '''

        # Start tracking metrics for this temporal split if not already.
        t_index = evaluation["t_index"]
        if t_index not in self._best_models:
            self._best_models[t_index] = {
                metric: (0.0, None)
                for metric in self._optimizing_metrics
            }
        # Check whether this model outperforms others on any metric.
        metrics_outperformed = []
        for metric in self._optimizing_metrics:
            this_score = evaluation[metric]
            best_score = self._best_models[t_index][metric][0]
            if this_score > best_score:
                self._best_models[t_index][metric] = \
                    [this_score, classifier, evaluation]
                metrics_outperformed.append(metric)
        # If it does, calculate the curves corresponding to this model.
        if len(metrics_outperformed) > 0:
            curves = self._calculate_curves(y_valid, y_proba, t_index)
            for metric in metrics_outperformed:
                self._best_models[t_index][metric].append(curves)


    def _write_evaluations(self, evaluations, path_to_eval=None):

        '''
        Save evaluation to CSV file.

        evaluations (list): collection of metrics across thresholds.
        path_to_eval (str): location at which to find CSV file of evaluations.

        '''

        # Determine whether to write header row per CSV file existance.
        if not path_to_eval:
            path_to_eval = self._ev_path
        write_mode = "w"
        if os.path.exists(path_to_eval):
            write_mode = "a"
        with open(path_to_eval, write_mode, newline="") as f:
            fieldnames = pbk._DEFAULT_EVALUATION_COLUMNS
            evaluation_writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Write header row.
            if write_mode == "w":
                evaluation_writer.writeheader()
            # Write all other rows.
            evaluation_writer.writerows(evaluations)
    

    def _calculate_curves(self, y_valid, y_proba, t_index):

        '''
        Calculate precision, recall, receiver operating characteristic curves
        and area under roc curve.

        y_valid (array): validation set of target variable.
        y_proba (array): classification score set of target variable.
        t_index (int): temporal set index.

        Return auc (float), precision (Series), recall (Series), roc (Series).

        '''

        suffix =  " (t=" + str(t_index) + ")"
        # Calculate precision and recall curves as Series.
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_true=y_valid,
            probas_pred=y_proba
        )
        thresholds = np.flip(np.insert(thresholds, len(thresholds), 1))
        precision_curve = pd.Series(
            precision,
            index=thresholds,
            name="Precision" + suffix
        )
        recall_curve = pd.Series(
            recall,
            index=thresholds,
            name="Recall" + suffix
        )
        # Calculate roc curve as Series.
        false_positive_rate, true_positive_rate, _ = metrics.roc_curve(
            y_true=y_valid,
            y_score=y_proba
        )
        roc_auc = np.round(metrics.roc_auc_score(y_valid, y_proba), 2)
        roc_curve = pd.Series(
            true_positive_rate,
            index=false_positive_rate,
            name="ROC (AUC=" + str(roc_auc) + ")" + suffix
        )
        return precision_curve, recall_curve, roc_curve
    

    def _unpack_best_models(self):

        '''
        Unpack the best models, plot their curves, and return their classifers.

        Return the best models (set of sklearn classifier objects).

        '''

        best_models = set()
        for t_index in self._best_models:
            for metric in self._best_models[t_index]:
                optimal_model = self._best_models[t_index][metric]
                _, classifier, evaluation, curves = optimal_model
                # Ensure the uniqueness of this model.
                if (t_index, classifier) not in best_models:
                    prefix = (
                        " ".join(evaluation["method"].split(" ")).title() +
                        " (t=" + str(t_index) + "," + 
                        " p=" + str(evaluation["p_index"]) + "): "
                    )
                    # Plot precision and recall curves.
                    pbp._plot_curves(
                        curves=curves[0] + curves[1],
                        title=prefix + "Precision and Recall",
                        x_axis="Threshold",
                        y_axis="Rate"
                    )
                    # Plot roc curve.
                    pbp._plot_curves(
                        curves=curves[2],
                        title=prefix + "Receiver Operating Characteristic",
                        x_axis="False Positive Rate",
                        y_axis="True Positive Rate"
                    )
                    # Add the model to the set of best models.
                    best_models.add((t_index, classifier))
        return best_models
    
    
    def execute_sql(self, query):

        # Connect to the database and execute the query.
        self._db_open()
        data = pd.read_sql(sql=query, con=self._db_con)
        self._db_close()
        # Coerce any value that could be a number as numeric.
        for variable in data.columns:
            data[variable] = pd.to_numeric(data[variable], errors="ignore")
        return data

