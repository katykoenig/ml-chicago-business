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
import chicago_storefronts_db as csdb


def build_storefronts_database(arguments):

    db = csdb.Storefronts()
    db.open()
    if arguments.download:
        db.create_licences_table()
        db.populate_licenses_table()
    db.create_storefronts_table(
        l_bound=np.datetime64(arguments.l_bound),
        u_bound=np.datetime64(arguments.u_bound)
    )
    db.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create the Chicago Storefronts database."
    )
    parser.add_argument(
        "--download",
        action="store_const",
        const=True,
        default=False,
        help="Rerequest business license data from the Chicago Data Portal.",
        dest="download"
    )
    parser.add_argument(
        "--lower",
        default=True,
        help="Indicate the earliest date from which to build a table.",
        dest="l_bound"
    )
    parser.add_argument(
        "--upper",
        default=True,
        help="Indicate the latest date from which to build a table.",
        dest="u_bound"
    )
    arguments = parser.parse_args()
    build_storefronts_database(arguments)
