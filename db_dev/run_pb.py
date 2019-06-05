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

    # Load pipeline and set methods set.
    ce = cepb.Plumbum("chicago entrepreneurship")
    ce.methods = cepbk.select_methods
    # Set temporal splits.
    tl_bound = np.datetime64(arguments.tl_bound)
    tu_bound = np.datetime64(arguments.tu_bound)
    vl_bound = np.datetime64(arguments.vl_bound)
    vu_bound = np.datetime64(arguments.vu_bound)
    temporal_splits = (tl_bound, tu_bound, vl_bound, vu_bound)
    ce.temporal_splits = temporal_splits
    # Model on this temporal split.
    ce.classify()


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
    arguments = parser.parse_args()
    run_entrepreneurship_pipeline(arguments)
