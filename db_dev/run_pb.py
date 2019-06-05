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

import argparse
import numpy as np
import chicago_entrepreneurship_pb as cepb
import chicago_entrepreneurship_pb_constants as cepbk


def run_entrepreneurship_pipeline(arguments):

    # Load pipeline and set methods and target variable.
    ce = cepb.Plumbum("chicago entrepreneurship")
    ce.methods = cepbk.select_methods
    ce.target_variable = "successful"
    # Set temporal splits.
    tl_bound = np.datetime64(arguments.tl_bound)
    tu_bound = np.datetime64(arguments.tu_bound)
    vl_bound = np.datetime64(arguments.vl_bound)
    vu_bound = np.datetime64(arguments.vu_bound)
    ce.temporal_splits = [(tl_bound, tu_bound, vl_bound, vu_bound)]
    # Model on this temporal split otherwise.
    if not arguments.skip_model:
        ce.classify()
    # Just create the datasets for the temporal split otherwise.
    else:
        ce._db_request(tl_bound, tu_bound, vl_bound, vu_bound, True)
        ce._db_request(tl_bound, tu_bound, vl_bound, vu_bound, False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the Chicago Entrepreneurship pipeline."
    )
    parser.add_argument(
        "--tl_bound",
        help="Temporal lower bound inclusive for training set.",
        dest="tl_bound"
    )
    parser.add_argument(
        "--tu_bound",
        help="Temporal upper bound exclusive for training set.",
        dest="tu_bound"
    )
    parser.add_argument(
        "--vl_bound",
        help="Temporal lower bound inclusive for validation set.",
        dest="vl_bound"
    )
    parser.add_argument(
        "--vu_bound",
        help="Temporal upper bound exclusive for validation set.",
        dest="vu_bound"
    )
    parser.add_argument(
        "--skip_model",
        action="store_const",
        const=True,
        default=False,
        help="Skip modeling and only return temporal sets.",
        dest="skip_model"
    )
    arguments = parser.parse_args()
    run_entrepreneurship_pipeline(arguments)
