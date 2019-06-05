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
import chicago_entrepreneurship_db as cedb


def build_entrepreneurship_database(arguments):

    db = cedb.Entrepreneurship()
    db.open()
    if arguments.blocks:
        db.create_blocks_table()
    if arguments.crimes:
        db.create_crimes_table()
    if arguments.licenses:
        db.create_licenses_table()
    if arguments.census:
        db.create_census_table(arguments.census)
    db.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create the Chicago Entrepreneurship database."
    )
    parser.add_argument(
        "--blocks",
        action="store_const",
        const=True,
        default=False,
        help="Rebuild the blocks table from the Chicago Data Portal API.",
        dest="blocks"
    )
    parser.add_argument(
        "--crimes",
        action="store_const",
        const=True,
        default=False,
        help="Rebuild the crimes table from the Chicago Data Portal API.",
        dest="crimes"
    )
    parser.add_argument(
        "--licenses",
        action="store_const",
        const=True,
        default=False,
        help="Rebuild the licenses table from the Chicago Data Portal API.",
        dest="licenses"
    )
    parser.add_argument(
        "--census",
        default=False,
        help="Rebuild the census table from the US Census Bureau ACS API.",
        dest="census"
    )
    arguments = parser.parse_args()
    build_entrepreneurship_database(arguments)
