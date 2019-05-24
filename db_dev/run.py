'''
Promoting Sustained Entrepreneurship in Chicago
Machine Learning for Public Policy
University of Chicago, CS & Harris School of Public Policy
May 2019

Rayid Ghani (@rayidghani)
Katy Koenig (@katykoenig)
Eric Langowski (@erhla)
Patrick Lavallee Delgado (@lavalleedelgado)

'''

import argparse
import chicago_storefronts_db as csdb


def run_chicago_storefronts_database(arguments):

    db = csdb.Storefronts()
    db.open()
    if arguments.no_download:
        db.create_storefronts_table()
        db.populate_storefronts_table()
    else:
        db.create_licences_table()
        db.populate_licenses_table()
        db.create_storefronts_table()
        db.populate_storefronts_table()
    db.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create the Chicago Storefronts database."
    )
    parser.add_argument(
        "--no_download",
        action="store_const",
        const=True,
        default=False,
        help="Create database without rerequesting from Chicago Data Portal API.",
        dest="no_download"
    )
    arguments = parser.parse_args()
    run_chicago_storefronts_database(arguments)
