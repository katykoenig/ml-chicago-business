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
import pandas as pd
import sqlite3

ACS_FIVE_YEAR = 2013    # American Community Survey Five Year edition.
JOIN_LIMIT = 60         # Maximum number of simultaneous joins.


def db_query(arguments):

    # Establish connection to the database.
    CHICAGO_DB = "chicago_entrepreneurship_db.sqlite3"
    db_con = sqlite3.connect(CHICAGO_DB)

    # Set temporal splits.
    train_lb = np.datetime64(arguments.train_lb)
    train_ub = np.datetime64(arguments.train_ub)
    valid_lb = np.datetime64(arguments.valid_lb)
    valid_ub = np.datetime64(arguments.valid_ub)
    buffer = np.timedelta64(2, "Y")
    # Create the datasets for this temporal split.
    temporal_select(db_con, train_lb, train_ub, valid_lb, valid_ub, buffer, \
        True, True)
    temporal_select(db_con, train_lb, train_ub, valid_lb, valid_ub, buffer, \
        False, True)


def temporal_select(db_con, train_lb, train_ub, valid_lb, valid_ub, buffer, \
    train, export_csv=False):

    '''
    Compile a temporal set from the database.

    db_con (Connection): database driver.
    train_lb (datetime): training set temporal lower bound inclusive.
    train_ub (datetime): training set temporal upper bound exclusive.
    valid_lb (datetime): validation set temporal lower bound inclusive.
    valid_ub (datetime): validation set temporal upper bound exclusive.
    buffer (timedelta): time between prediction and observation.
    train (bool): whether to request a training or validation set.
    export_csv (bool): whether to save the temporal set as a CSV file.

    Return data (DataFrame).

    '''

    # Convert np.datetime64 to pd.Timestamp for arithmetic convenience.
    train_lb = pd.Timestamp(train_lb)
    train_ub = pd.Timestamp(train_ub)
    valid_lb = pd.Timestamp(valid_lb)
    valid_ub = pd.Timestamp(valid_ub)
    # Toggle between bounds for training or validation set.
    if train:
        lb, ub = train_lb, train_ub
    else:
        lb, ub = valid_lb, valid_ub
    storefronts = request_storefronts(db_con, lb, ub, buffer)
    licenses = request_licenses(db_con, lb, ub, train_lb, train_ub)
    crimes = request_crimes(db_con, lb, ub, train_lb, train_ub)
    census = request_census(db_con)
    # Join storefronts, licenses, crimes, census data.
    data = storefronts.merge(licenses, on="sf_id").merge(crimes, on="block")
    data["block_group"] = data["block"].floordiv(1000)
    data = data.merge(census, on="block_group")
    # Save the dataframe to CSV if requested.
    if export_csv:
        if train:
            set_type = "train"
        else:
            set_type = "valid"
        filename = "_".join(
            [set_type, str(lb.date()), str(ub.date())]
        )
        data.to_csv(filename + ".csv", index=False)
    # Return the dataframe for modeling.
    return data


def request_storefronts(db_con, lb, ub, buffer):

    # Request and collect storefronts in this period.
    select_storefronts = f'''
    WITH 
        storefronts_general AS ( 
            SELECT 
                account_number || '-' || site_number AS sf_id, 
                MIN(date_issued) AS earliest_issue, 
                MAX(expiration_date) AS latest_issue 
            FROM licenses 
            WHERE DATETIME(expiration_date) >= DATETIME('{lb}') 
            AND DATETIME(date_issued) < DATETIME('{ub}') 
            GROUP BY account_number, site_number 
        ), 
        storefronts_general_future AS (
            SELECT 
                account_number || '-' || site_number AS sf_id 
            FROM licenses 
            WHERE DATETIME(expiration_date) >= DATETIME('{lb + buffer}') 
            AND DATETIME(date_issued) < DATETIME('{ub + buffer}') 
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
                police_district, 
                RANK() OVER ( 
                    PARTITION BY account_number, site_number 
                    ORDER BY date_issued DESC 
                ) AS last_location 
            FROM licenses 
            WHERE block IS NOT NULL 
            AND DATETIME(expiration_date) >= DATETIME('{lb}') 
            AND DATETIME(date_issued) < DATETIME('{ub}') 
        ), 
        storefronts_blocks AS ( 
            SELECT 
                COUNT( 
                    DISTINCT account_number || '-' || site_number 
                ) AS storefronts_on_block, 
                block 
            FROM licenses 
            WHERE block IS NOT NULL 
            AND DATETIME(expiration_date) >= DATETIME('{lb}') 
            AND DATETIME(date_issued) < DATETIME('{ub}') 
            GROUP BY block 
        ) 
    SELECT DISTINCT 
        sf_id, 
        earliest_issue, 
        latest_issue, 
        storefronts_on_block, 
        block, 
        police_district, 
        successful 
    FROM storefronts_success 
    JOIN storefronts_location USING (sf_id) 
    JOIN storefronts_blocks USING (block) 
    WHERE last_location = 1;
    '''
    return pd.read_sql(select_storefronts, db_con)


def request_licenses(db_con, lb, ub, train_lb, train_ub):

    # Initialize cursor.
    db_cursor = db_con.cursor()
    # Request licenses extant in training period by storefront.
    select_storefronts_licenses = f'''
    SELECT * 
    FROM ( 
        SELECT account_number || '-' || site_number AS sf_id 
        FROM licenses 
        WHERE DATETIME(expiration_date) >= DATETIME('{lb}') 
        AND DATETIME(date_issued) < DATETIME('{ub}') 
        GROUP BY account_number, site_number 
    ) AS storefronts 
    '''
    select_extant_licenses = db_cursor.execute(f'''
        SELECT DISTINCT license_code 
        FROM licenses 
        WHERE DATETIME(expiration_date) >= DATETIME('{train_lb}') 
        AND DATETIME(date_issued) < DATETIME('{train_ub}') 
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
                AND DATETIME(expiration_date) >= DATETIME('{lb}') 
                AND DATETIME(date_issued) < DATETIME('{ub}') 
                GROUP BY account_number, site_number 
            ) AS L_{license_lable} USING (sf_id) 
            '''
        ) 
        # Execute these joins at the join limit or with the last relation.
        if i % JOIN_LIMIT == 0 or i == len(extant_licenses) - 1:
            batch = pd.read_sql(
                select_storefronts_licenses + " ".join(licenses_join_queue),
                db_con
            )
            licenses_join_complete.append(batch.fillna(0))
            licenses_join_queue = []
    # Merge batches into one dataframe.
    licenses = licenses_join_complete.pop()
    for batch in licenses_join_complete:
        licenses = licenses.merge(batch, on="sf_id")
    return licenses


def request_crimes(db_con, lb, ub, train_lb, train_ub):

    # Initialize cursor.
    db_cursor = db_con.cursor()
    # Request crimes extant and one year prior in training period by block.
    crime_lb = lb - np.timedelta64(1, "Y")
    select_crimes_general = f'''
    WITH 
        domestic AS ( 
            SELECT block, COUNT(domestic) AS domestic_count 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{crime_lb}') 
            AND DATETIME(date) < DATETIME('{ub}') 
            AND domestic = 'True' 
            GROUP BY block 
        ), 
        arrest AS ( 
            SELECT block, COUNT(arrest) AS arrest_count 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{crime_lb}') 
            AND DATETIME(date) < DATETIME('{ub}') 
            AND arrest = 'True' 
            GROUP BY block 
        ), 
        sum AS (
            SELECT block, COUNT(primary_type) AS crime_count 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{crime_lb}') 
            AND DATETIME(date) < DATETIME('{ub}') 
            GROUP BY block 
        ) 
    SELECT
        block, 
        domestic_count, 
        arrest_count, 
        crime_count 
    FROM blocks 
    LEFT JOIN domestic USING (block) 
    LEFT JOIN arrest USING (block) 
    LEFT JOIN sum USING (block);
    '''
    crime_general = pd.read_sql(select_crimes_general, db_con)
    select_crimes_blocks = f'''
    SELECT * 
    FROM (
        SELECT DISTINCT block 
        FROM blocks
    ) AS blocks 
    '''
    select_extant_crimes = db_cursor.execute(f'''
        SELECT DISTINCT primary_type 
        FROM crimes 
        WHERE DATETIME(date) >= DATETIME('{train_lb - np.timedelta64(1, "Y")}') 
        AND DATETIME(date) < DATETIME('{train_ub}');
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
                COUNT(primary_type) AS crime_{crime_label} 
            FROM crimes 
            WHERE primary_type = '{extant_crime[0]}' 
            AND DATETIME(date) >= DATETIME('{crime_lb}') 
            AND DATETIME(date) < DATETIME('{ub}') 
            GROUP BY block 
            ) AS C_{crime_label} USING (block) 
            '''
        )
        # Execute these joins at the join limit or with the last relation.
        if i % JOIN_LIMIT == 0 or i == len(extant_crimes) - 1:
            batch = pd.read_sql(
                select_crimes_blocks + " ".join(crimes_join_queue),
                db_con
            )
            crimes_join_complete.append(batch.fillna(0))
            crimes_join_queue = []
    # Merge batches into one dataframe.
    crimes = crimes_join_complete.pop()
    for batch in crimes_join_complete:
        crimes = crimes.merge(batch, on="block")
    # Convert counts to daily averages.
    time_delta = (ub - crime_lb) / np.timedelta64(365, "D")
    crimes_avg = crimes.drop(columns="block").div(time_delta)
    crimes_avg.columns = [
        "avg_" + str(column)
        for column in crimes_avg.columns
    ]
    return pd.concat([crimes["block"], crimes_avg], axis=1).fillna(0)


def request_census(db_con):

    # Request and aggregate census data.
    select_census = f"SELECT * FROM census WHERE end_year = '{ACS_FIVE_YEAR}';"
    census = pd.read_sql(select_census, db_con)
    census["total_population"] = census["total_male"] + census["total_female"]
    aggregations = {
        "pct_male_children": ("total_male", [
            "total_male_lt_5",
            "total_male_5_9",
            "total_male_10_14",
            "total_male_15_17"
        ]),
        "pct_male_working": ("total_male", [
            "total_male_18_19",
            "total_male_20",
            "total_male_21",
            "total_male_22_24",
            "total_male_25_29",
            "total_male_30_34",
            "total_male_35_39",
            "total_male_40_44",
            "total_male_45_49",
            "total_male_50_54",
            "total_male_55_59",
            "total_male_60_61",
            "total_male_62_64"
        ]),
        "pct_male_elderly": ("total_male", [
            "total_male_65_66",
            "total_male_67_69",
            "total_male_70_74",
            "total_male_75_79",
            "total_male_80_84",
            "total_male_gte_85"
        ]),
        "pct_female_children": ("total_female", [
            "total_female_lt_5",
            "total_female_5_9",
            "total_female_10_14",
            "total_female_15_17"
        ]),
        "pct_female_working": ("total_female", [
            "total_female_18_19",
            "total_female_20",
            "total_female_21",
            "total_female_22_24",
            "total_female_25_29",
            "total_female_30_34",
            "total_female_35_39",
            "total_female_40_44",
            "total_female_45_49",
            "total_female_50_54",
            "total_female_55_59",
            "total_female_60_61",
            "total_female_62_64"
        ]),
        "pct_female_elderly": ("total_female", [
            "total_female_65_66",
            "total_female_67_69",
            "total_female_70_74",
            "total_female_75_79",
            "total_female_80_84",
            "total_female_gte_85"
        ]),
        "pct_low_travel_time": ("total_commute_time", [
            "total_commute_lt_5",
            "total_commute_5_9",
            "total_commute_10_14",
            "total_commute_15_19",
            "total_commute_20_24"
        ]),
        "pct_medium_travel_time": ("total_commute_time", [
            "total_commute_25_29",
            "total_commute_30_34",
            "total_commute_35_39",
            "total_commute_40_44",
            "total_commute_45_59"
        ]),
        "pct_high_travel_time": ("total_commute_time", [
            "total_commute_60_89",
            "total_commute_gte_90"
        ]),
        "pct_below_poverty": ("hhinc_respondents", [
            "hhinc_00_10K",
            "hhinc_10_15K",
            "hhinc_15_20K",
            "hhinc_20_25K"
        ]),
        "pct_below_median_income": ("hhinc_respondents", [
            "hhinc_25_30K",
            "hhinc_30_35K",
            "hhinc_35_40K",
            "hhinc_40_45K",
            "hhinc_45_50K",
            "hhinc_50_60K"
        ]),
        "pct_above_median_income": ("hhinc_respondents", [
            "hhinc_60_75K",
            "hhinc_75_100K"
        ]),
        "pct_high_income": ("hhinc_respondents", [
            "hhinc_100_125K",
            "hhinc_125_150K",
            "hhinc_150_200K"
        ]),
        "pct_white": ("race_respondents", ["race_white"]),
        "pct_black": ("race_respondents", ["race_black"]),
        "pct_asian": ("race_respondents", ["race_asian"]),
        "pct_hispanic": ("race_respondents", ["race_hispanic"])
    }
    columns_to_drop = ["start_year", "end_year"]
    for aggregation, columns in aggregations.items():
        denominator, numerators = columns
        census[aggregation] = census[numerators] \
            .agg("sum", axis=1) \
            .div(census[denominator]) \
            .fillna(0)
        columns_to_drop.extend(numerators)
        columns_to_drop.append(denominator)
    return census.drop(columns=list(set(columns_to_drop)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the Chicago Entrepreneurship database."
    )
    parser.add_argument(
        "--train_lb",
        help="Temporal lower bound inclusive for training set.",
        dest="train_lb"
    )
    parser.add_argument(
        "--train_ub",
        help="Temporal upper bound exclusive for training set.",
        dest="train_ub"
    )
    parser.add_argument(
        "--valid_lb",
        help="Temporal lower bound inclusive for validation set.",
        dest="valid_lb"
    )
    parser.add_argument(
        "--valid_ub",
        help="Temporal upper bound exclusive for validation set.",
        dest="valid_ub"
    )
    arguments = parser.parse_args()
    db_query(arguments)
