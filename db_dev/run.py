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
    if arguments.create_licenses_table:
        db.create_licences_table()
    if arguments.populate_licenses_table:
        db.populate_licenses_table()
    if arguments.create_storefronts_table:
        db.create_storefronts_table()
    if arguments.populate_storefronts_table:
        db.create_storefronts_table()
        db.populate_storefronts_table()
    db.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create the Chicago Storefronts database."
    )
    parser.add_argument(
        "--cl",
        action="store_const",
        const=True,
        default=False,
        help="Create the licenses table.",
        dest="create_licenses_table"
    )
    parser.add_argument(
        "--pl",
        action="store_const",
        const=True,
        default=False,
        help="Populate the licenses table.",
        dest="populate_licenses_table"
    )
    parser.add_argument(
        "--cs",
        action="store_const",
        const=True,
        default=False,
        help="Create the storefronts table.",
        dest="create_storefronts_table"
    )
    parser.add_argument(
        "--ps",
        action="store_const",
        const=True,
        default=False,
        help="Populate the storefronts table.",
        dest="populate_storefronts_table"
    )
    arguments = parser.parse_args()
    run_chicago_storefronts_database(arguments)
