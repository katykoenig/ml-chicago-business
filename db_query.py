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

    # Set join limit constant and initialize connection cursor.
    JOIN_LIMIT = 60
    db_cursor = db_con.cursor()
    # Convert np.datetime64 to pd.Timestamp for convenience.
    train_lb = pd.Timestamp(train_lb)
    train_ub = pd.Timestamp(train_ub)
    valid_lb = pd.Timestamp(valid_lb)
    valid_ub = pd.Timestamp(valid_ub)
    # Toggle between bounds for training or validation set.
    if train:
        lb, ub = train_lb, train_ub
    else:
        lb, ub = valid_lb, valid_ub
    # Request and collect storefronts in this period.
    select_storefronts = f'''
    WITH 
        storefronts_general AS ( 
            SELECT 
                account_number || '-' || site_number AS sf_id, 
                MIN(issue_date) AS earliest_issue, 
                MAX(expiry_date) AS latest_issue 
            FROM licenses 
            WHERE DATETIME(expiry_date) >= DATETIME('{lb}') 
            AND DATETIME(issue_date) < DATETIME('{ub}') 
            GROUP BY account_number, site_number 
        ), 
        storefronts_general_future AS (
            SELECT 
                account_number || '-' || site_number AS sf_id 
            FROM licenses 
            WHERE DATETIME(expiry_date) >= DATETIME('{lb + buffer}') 
            AND DATETIME(issue_date) < DATETIME('{ub + buffer}') 
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
                    ORDER BY issue_date DESC 
                ) AS last_location 
            FROM licenses 
            WHERE block IS NOT NULL 
            AND DATETIME(expiry_date) >= DATETIME('{lb}') 
            AND DATETIME(issue_date) < DATETIME('{ub}') 
        ), 
        storefronts_blocks AS ( 
            SELECT 
                COUNT( 
                    DISTINCT account_number || '-' || site_number 
                ) AS storefronts_on_block, 
                block 
            FROM licenses 
            WHERE block IS NOT NULL 
            AND DATETIME(expiry_date) >= DATETIME('{lb}') 
            AND DATETIME(issue_date) < DATETIME('{ub}') 
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
    storefronts = pd.read_sql(select_storefronts, db_con)
    # Request and collect licenses extant in training period by storefront.
    select_storefronts_licenses = f'''
    SELECT * 
    FROM ( 
        SELECT account_number || '-' || site_number AS sf_id 
        FROM licenses 
        WHERE DATETIME(expiry_date) >= DATETIME('{lb}') 
        AND DATETIME(issue_date) < DATETIME('{ub}') 
        GROUP BY account_number, site_number 
    ) AS storefronts 
    '''
    select_extant_licenses = db_cursor.execute(f'''
        SELECT DISTINCT license_code 
        FROM licenses 
        WHERE DATETIME(expiry_date) >= DATETIME('{train_lb}') 
        AND DATETIME(issue_date) < DATETIME('{train_ub}') 
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
                AND DATETIME(expiry_date) >= DATETIME('{lb}') 
                AND DATETIME(issue_date) < DATETIME('{ub}') 
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
    # Request and collect crimes extant in training period by block.
    crime_lb = lb - np.timedelta64(1, "Y")
    select_crime_general = f'''
    WITH 
        domestic AS ( 
            SELECT block, COUNT(domestic) AS domestic_count 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{crime_lb}') 
            AND DATETIME(date) < DATETIME('{train_ub}') 
            AND domestic = 'True' 
            GROUP BY block 
        ), 
        arrest AS ( 
            SELECT block, COUNT(arrest) AS arrest_count 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{crime_lb}') 
            AND DATETIME(date) < DATETIME('{train_ub}') 
            AND arrest = 'True' 
            GROUP BY block 
        ), 
        sum AS (
            SELECT block, COUNT(crime) AS crime_count 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{crime_lb}') 
            AND DATETIME(date) < DATETIME('{train_ub}') 
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
    crime_general = pd.read_sql(select_crime_general, db_con)
    select_crimes_blocks = f'''
    SELECT * 
    FROM (
        SELECT DISTINCT block 
        FROM blocks
    ) AS blocks 
    '''
    select_extant_crimes = db_cursor.execute(f'''
        SELECT DISTINCT crime 
        FROM crimes 
        WHERE DATETIME(date) >= DATETIME('{crime_lb}') 
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
                COUNT(crime) AS crime_{crime_label} 
            FROM crimes 
            WHERE crime = '{extant_crime[0]}' 
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
    # # Convert counts to annual averages.
    # time_delta = (ub - crime_lb) / np.timedelta64(1, "Y")
    # for column in crimes.columns:
    #     if column == "block":
    #         continue
    #     crimes[column] = crimes[column].div(time_delta).fillna(0).round()
    # Request and aggregate census data.
    census = pd.read_sql("SELECT * FROM census", db_con)
    census["total_population"] = census["total_male"] + census["total_female"]
    aggregations = {
        "pct_male_children": ("total_male", [
            "total_male_<5",
            "total_male_5-9",
            "total_male_10-14",
            "total_male_15-17"
        ]),
        "pct_male_working": ("total_male", [
            "total_male_18-19",
            "total_male_20",
            "total_male_21",
            "total_male_22-24",
            "total_male_25-29",
            "total_male_30-34",
            "total_male_35-39",
            "total_male_40-44",
            "total_male_45-49",
            "total_male_50-54",
            "total_male_55-59",
            "total_male_60-61",
            "total_male_62-64"
        ]),
        "pct_male_elderly": ("total_male", [
            "total_male_65-66",
            "total_male_67-69",
            "total_male_70-74",
            "total_male_75-79",
            "total_male_80-84",
            "total_male_>=85"
        ]),
        "pct_female_children": ("total_female", [
            "total_female_<5",
            "total_female_5-9",
            "total_female_10-14",
            "total_female_15-17"
        ]),
        "pct_female_working": ("total_female", [
            "total_female_18-19",
            "total_female_20",
            "total_female_21",
            "total_female_22-24",
            "total_female_25-29",
            "total_female_30-34",
            "total_female_35-39",
            "total_female_40-44",
            "total_female_45-49",
            "total_female_50-54",
            "total_female_55-59",
            "total_female_60-61",
            "total_female_62-64"
        ]),
        "pct_female_elderly": ("total_female", [
            "total_female_65-66",
            "total_female_67-69",
            "total_female_70-74",
            "total_female_75-79",
            "total_female_80-84",
            "total_female_>=85"
        ]),
        "pct_low_travel_time": ("total_commute_time", [
            "total_commute_<5",
            "total_commute_5-9",
            "total_commute_10-14",
            "total_commute_15-19",
            "total_commute_20-24"
        ]),
        "pct_medium_travel_time": ("total_commute_time", [
            "total_commute_25-29",
            "total_commute_30-34",
            "total_commute_35-39",
            "total_commute_40-44",
            "total_commute_45-59"
        ]),
        "pct_high_travel_time": ("total_commute_time", [
            "total_commute_60-89",
            "total_commute_>=90"
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
    columns_to_drop = []
    for aggregation, columns in aggregations.items():
        denominator, numerators = columns
        census[aggregation] = census[numerators] \
            .agg("sum", axis=1) \
            .div(census[denominator]) \
            .fillna(0)
        columns_to_drop.extend(numerators)
        columns_to_drop.append(denominator)
    census = census.drop(columns=list(set(columns_to_drop)))
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the Chicago Entrepreneurship pipeline."
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
