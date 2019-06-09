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


# Establish connection to the database.
CHICAGO_DB = "chicago_entrepreneurship_db.sqlite3"
db = sqlite3.connect(CHICAGO_DB)


def db_query(arguments):

    # Set temporal splits.
    train_lb = np.datetime64(arguments.train_lb)
    train_ub = np.datetime64(arguments.train_ub)
    valid_lb = np.datetime64(arguments.valid_lb)
    valid_ub = np.datetime64(arguments.valid_ub)
    interval = np.timedelta64(2, "Y")
    # Create the datasets for this temporal split.
    temporal_select(train_lb, train_ub, valid_lb, valid_ub, interval, True, \
        True)
    temporal_select(train_lb, train_ub, valid_lb, valid_ub, interval, False, \
        True)


def temporal_select(train_lb, train_ub, valid_lb, valid_ub, interval, train, \
    export_csv=False):

    '''
    Compile a temporal set from the database.

    train_lb (datetime): temporal lower bound inclusive for training set.
    train_ub (datetime): temporal upper bound exclusive for training set.
    valid_lb (datetime): temporal lower bound inclusive for validation set.
    valid_ub (datetime): temporal upper bound exclusive for validation set.
    interval (timedelta): length of time between prediction and observation.
    train (bool): whether to request a training or validation set.
    export_csv (bool): whether to also save the temporal set as a CSV file.

    Return data (DataFrame).

    '''

    # Set join limit constant and initialize connection cursor.
    JOIN_LIMIT = 60
    db_cursor = db.cursor()
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
            WHERE DATETIME(expiry_date) >= DATETIME('{lb + interval}') 
            AND DATETIME(issue_date) < DATETIME('{ub + interval}') 
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
        successful 
    FROM storefronts_success 
    JOIN storefronts_location USING (sf_id) 
    JOIN storefronts_blocks USING (block) 
    WHERE last_location = 1;
    '''
    storefronts = pd.read_sql(select_storefronts, db)
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
                db
            )
            licenses_join_complete.append(batch.fillna(0))
            licenses_join_queue = []
    # Merge batches into one dataframe.
    licenses = licenses_join_complete.pop()
    for batch in licenses_join_complete:
        licenses = licenses.merge(batch, on="sf_id")
    # Request and collect crimes extant in training period by block.
    select_crime_general = f'''
    WITH 
        domestic AS ( 
            SELECT block, COUNT(domestic) AS domestic_sum 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{train_lb}') 
            AND DATETIME(date) < DATETIME('{train_ub}') 
            AND domestic = 'True' 
            GROUP BY block 
        ), 
        arrest AS ( 
            SELECT block, COUNT(arrest) AS arrest_sum 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{train_lb}') 
            AND DATETIME(date) < DATETIME('{train_ub}') 
            AND arrest = 'True' 
            GROUP BY block 
        ),
        sum AS (
            SELECT block, COUNT(crime) AS crime_sum 
            FROM crimes 
            WHERE DATETIME(date) >= DATETIME('{train_lb}') 
            AND DATETIME(date) < DATETIME('{train_ub}') 
            GROUP BY block 
        ) 
    SELECT DISTINCT block 
    FROM blocks 
    LEFT JOIN domestic USING (block) 
    LEFT JOIN arrest USING (block) 
    LEFT JOIN sum USING (block);
    '''
    crime_general = pd.read_sql(select_crime_general, db)
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
        WHERE DATETIME(date) >= DATETIME('{train_lb}') 
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
            AND DATETIME(date) >= DATETIME('{lb}') 
            AND DATETIME(date) < DATETIME('{ub}') 
            GROUP BY block 
            ) AS C_{crime_label} USING (block) 
            '''
        )
        # Execute these joins at the join limit or with the last relation.
        if i % JOIN_LIMIT == 0 or i == len(extant_crimes) - 1:
            batch = pd.read_sql(
                select_crimes_blocks + " ".join(crimes_join_queue),
                db
            )
            crimes_join_complete.append(batch.fillna(0))
            crimes_join_queue = []
    # Merge batches into one dataframe.
    crimes = crimes_join_complete.pop()
    for batch in crimes_join_complete:
        crimes = crimes.merge(batch, on="block")
    # Request census data.
    census = pd.read_sql("SELECT * FROM census", db)
    # Join storefronts, licenses, crimes, census data.
    data = storefronts.merge(licenses, on="sf_id").merge(crimes, on="block")
    data["block_group"] = data["block"].apply(lambda x: x // 1000)
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
