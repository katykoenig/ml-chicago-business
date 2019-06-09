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
import pb_classify as pb
import pb_constants as pbk
from db_query_copy import temporal_select


def pb_run():

    # Load pipeline and set methods and target variable.
    ce = pb.Plumbum("chicago entrepreneurship")
    ce.methods = pbk.short_methods
    ce.target_variable = "successful"
    # Set temporal splits and select function.
    ce.temporal_splits = [
        (
            pd.Timestamp("2010-01-01"), pd.Timestamp("2015-06-01"),
            pd.Timestamp("2017-05-31"), pd.Timestamp("2017-06-01")
        )
    ]
    ce.temporal_select = temporal_select
    # Run classification models.
    ce.classify()

if __name__ == "__main__":

    pb_run()
