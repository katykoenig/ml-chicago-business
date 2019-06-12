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

import os
import argparse
import ast
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
import sqlite3


PWD = os.path.dirname(__file__)
CHICAGO_DB = os.path.join(PWD, "chicago_entrepreneurship_db.sqlite3")
BLOCKS_API = "https://data.cityofchicago.org/resource/bt9m-d2mf.json"
BLOCKS_COLUMNS = (
    "the_geom",
    "geoid10"
)
CRIMES_API = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
CRIMES_COLUMNS = (
    "date",
    "primary_type",
    "arrest",
    "domestic",
    "latitude",
    "longitude"
)
LICENSES_API = "https://data.cityofchicago.org/resource/xqx5-8hwx.json"
LICENSES_COLUMNS = (
    "account_number",
    "site_number",
    "license_code",
    "date_issued",
    "expiration_date",
    "police_district",
    "latitude",
    "longitude"
)
CENSUS_API = ("https://api.census.gov/data/", "/acs/acs5?")
CENSUS_COLUMNS = {
    "B01001_002E": "total_male",
    "B01001_003E": "total_male_lt_5",
    "B01001_004E": "total_male_5_9",
    "B01001_005E": "total_male_10_14",
    "B01001_006E": "total_male_15_17",
    "B01001_007E": "total_male_18_19",
    "B01001_008E": "total_male_20",
    "B01001_009E": "total_male_21",
    "B01001_010E": "total_male_22_24",
    "B01001_011E": "total_male_25_29",
    "B01001_012E": "total_male_30_34",
    "B01001_013E": "total_male_35_39",
    "B01001_014E": "total_male_40_44",
    "B01001_015E": "total_male_45_49",
    "B01001_016E": "total_male_50_54",
    "B01001_017E": "total_male_55_59",
    "B01001_018E": "total_male_60_61",
    "B01001_019E": "total_male_62_64",
    "B01001_020E": "total_male_65_66",
    "B01001_021E": "total_male_67_69",
    "B01001_022E": "total_male_70_74",
    "B01001_023E": "total_male_75_79",
    "B01001_024E": "total_male_80_84",
    "B01001_025E": "total_male_gte_85",
    "B01001_026E": "total_female",
    "B01001_027E": "total_female_lt_5",
    "B01001_028E": "total_female_5_9",
    "B01001_029E": "total_female_10_14",
    "B01001_030E": "total_female_15_17",
    "B01001_031E": "total_female_18_19",
    "B01001_032E": "total_female_20",
    "B01001_033E": "total_female_21",
    "B01001_034E": "total_female_22_24",
    "B01001_035E": "total_female_25_29",
    "B01001_036E": "total_female_30_34",
    "B01001_037E": "total_female_35_39",
    "B01001_038E": "total_female_40_44",
    "B01001_039E": "total_female_45_49",
    "B01001_040E": "total_female_50_54",
    "B01001_041E": "total_female_55_59",
    "B01001_042E": "total_female_60_61",
    "B01001_043E": "total_female_62_64",
    "B01001_044E": "total_female_65_66",
    "B01001_045E": "total_female_67_69",
    "B01001_046E": "total_female_70_74",
    "B01001_047E": "total_female_75_79",
    "B01001_048E": "total_female_80_84",
    "B01001_049E": "total_female_gte_85",
    "B08303_001E": "total_commute_time",
    "B08303_002E": "total_commute_lt_5",
    "B08303_003E": "total_commute_5_9",
    "B08303_004E": "total_commute_10_14",
    "B08303_005E": "total_commute_15_19",
    "B08303_006E": "total_commute_20_24",
    "B08303_007E": "total_commute_25_29",
    "B08303_008E": "total_commute_30_34",
    "B08303_009E": "total_commute_35_39",
    "B08303_010E": "total_commute_40_44",
    "B08303_011E": "total_commute_45_59",
    "B08303_012E": "total_commute_60_89",
    "B08303_013E": "total_commute_gte_90",
    "B15012_001E": "total_bachelors_degrees",
    "B03002_001E": "race_respondents",
    "B03002_003E": "race_white",
    "B03002_004E": "race_black",
    "B03002_006E": "race_asian",
    "B03002_012E": "race_hispanic",
    "B19001_001E": "hhinc_respondents",
    "B19001_002E": "hhinc_00_10K",
    "B19001_003E": "hhinc_10_15K",
    "B19001_004E": "hhinc_15_20K",
    "B19001_005E": "hhinc_20_25K",
    "B19001_006E": "hhinc_25_30K",
    "B19001_007E": "hhinc_30_35K",
    "B19001_008E": "hhinc_35_40K",
    "B19001_009E": "hhinc_40_45K",
    "B19001_010E": "hhinc_45_50K",
    "B19001_011E": "hhinc_50_60K",
    "B19001_012E": "hhinc_60_75K",
    "B19001_013E": "hhinc_75_100K",
    "B19001_014E": "hhinc_100_125K",
    "B19001_015E": "hhinc_125_150K",
    "B19001_016E": "hhinc_150_200K"
}

def db_build(arguments):

    db = Entrepreneurship()
    db.open()
    if arguments.blocks:
        db.download_chicago_data(BLOCKS_API, BLOCKS_COLUMNS, "blocks")
    if arguments.crimes:
        db.download_chicago_data(CRIMES_API, CRIMES_COLUMNS, "crimes")
    if arguments.licenses:
        db.download_chicago_data(LICENSES_API, LICENSES_COLUMNS, "licenses")
    if arguments.census:
        db.download_census_data(CENSUS_API, CENSUS_COLUMNS, arguments.census)
    db.close()


class Entrepreneurship:

    def __init__(self):

        self.dbname = "Chicago Entrepreneurship"


    def open(self):

        self.db_con = sqlite3.connect(CHICAGO_DB)
        self.db_cur = self.db_con.cursor()
    

    def close(self):

        self.db_con.close()
    

    def download_chicago_data(self, api, columns, table):

        record_batches = []
        record_count = 0
        record_queue = 1000
        # Request data from the Chicago Data Portal.
        while record_count % record_queue == 0:
            request = requests.get(
                api,
                params={
                    "$select": ", ".join(columns),
                    "$offset": record_count,
                    "$limit": record_queue
                }
            )
            new_records = request.json()
            record_count += len(new_records)
            record_batches.extend(new_records)
        records = pd.DataFrame(record_batches)
        if table == "crimes" or table == "licenses":
            # Drop observations with missing geographic indicators.
            records = records[records["latitude"].notna()]
            records = records[records["longitude"].notna()]
            records = records[records["latitude"] != 'None']
            records = records[records["longitude"] != 'None']
            # Set geometry for spatial join.
            records["the_geom"] = records.apply(
                lambda x: shapely.geometry.Point(
                    ast.literal_eval(x["longitude"]),
                    ast.literal_eval(x["latitude"])
                ),
                axis=1
            )
            records = gpd.GeoDataFrame(records).set_geometry("the_geom")
            # Collect census blocks and set geometry.
            get_blocks = "SELECT * FROM blocks;"
            blocks = pd.read_sql(get_blocks, self.db_con)
            blocks["the_geom"] = blocks["the_geom"].apply(
                lambda x: shapely.geometry.shape(ast.literal_eval(x))
            )
            blocks = gpd.GeoDataFrame(blocks).set_geometry("the_geom")
            # Join records and blocks on intersection of their geometry.
            columns = list(columns) + ["block"]
            records = gpd.sjoin(records, blocks)
            records = records[columns]
        for column in records.columns:
            records[column] = records[column].astype(str)
        records.to_sql(table, self.db_con, if_exists="replace", index=False)


    def download_census_data(self, api, columns, year):

        census_batches = []
        census_variables = list(columns.keys())
        census_queue = 50
        # Request data from the US Census Bureau.
        while census_variables:
            request = requests.get(
                api[0] + str(year) + api[1],
                params={
                    "get": ",".join(census_variables[:census_queue]),
                    "for": "block group:*",
                    "in": "state:17 county:031"
                }
            )
            # Load columns as a dataframe and infer headers from the first row.
            new_columns = pd.DataFrame(request.json())
            new_columns.columns = new_columns.iloc[0]
            new_columns = new_columns.drop(0)
            # Concatenate location attributes to form block group string.
            new_columns["block_group"] = new_columns \
                .iloc[:, [-4, -3, -2, -1]] \
                .apply(lambda x: "".join(x.values.astype(str)), axis=1)
            new_columns = new_columns.drop(new_columns.columns[-5:][:4], axis=1)
            # Update census variables in queue and save new columns.
            census_variables = census_variables[census_queue:]
            census_batches.append(new_columns)
        records = census_batches.pop()
        for batch in census_batches:
            records = records.merge(batch, on="block_group")
        records.columns = [
            CENSUS_COLUMNS.get(column, column)
            for column in records.columns
        ]
        records["ACS_year"] = year
        records.to_sql("census", self.db_con, if_exists="replace", index=False)


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
    db_build(arguments)
