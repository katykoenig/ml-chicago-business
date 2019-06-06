'''
Promoting Sustained Entrepreneurship in Chicago
Machine Learning for Public Policy
University of Chicago, CS & Harris School of Public Policy
June 2019

Rayid Ghani (@rayidghani)
Katy Koenig (@katykoenig)
Eric Langowski (@erhla)
Patrick Lavallee Delgado (@lavalleedelgado)

'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import chicago_entrepreneurship_pb as cepb
import chicago_entrepreneurship_pb_constants as cepbk


def run_entrepreneurship_pipeline():

    # Load pipeline and set methods and target variable.
    ce = cepb.Plumbum("chicago entrepreneurship")
    ce.target_variable = "successful"
    ce.methods = {"random_forest": (
        RandomForestClassifier(),
        {
            "criterion": ["entropy"],
            "max_depth": [10],
            "max_features": ["sqrt"],
            "min_samples_split": [2, 5, 10],
            "n_estimators": [100, 1000],
            "n_jobs": [-1],
            "random_state": [0]
        }
    )}
    # Set temporal splits.
    ce.temporal_splits = [
        (
            pd.Timestamp("2010-01-01"), pd.Timestamp("2015-06-01"),
            pd.Timestamp("2017-05-31"), pd.Timestamp("2017-06-01")
        ),
        (
            pd.Timestamp("2010-01-01"), pd.Timestamp("2014-06-01"),
            pd.Timestamp("2016-05-31"), pd.Timestamp("2016-06-01")
        ),
        (
            pd.Timestamp("2010-01-01"), pd.Timestamp("2013-06-01"),
            pd.Timestamp("2015-05-31"), pd.Timestamp("2015-06-01")
        ),
        (
            pd.Timestamp("2010-01-01"), pd.Timestamp("2012-06-01"),
            pd.Timestamp("2014-05-31"), pd.Timestamp("2014-06-01")
        )
    ]
    # Run classification models.
    ce.classify()


if __name__ == "__main__":

    run_entrepreneurship_pipeline()
