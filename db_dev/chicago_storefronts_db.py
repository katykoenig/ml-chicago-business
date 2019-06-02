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
import pandas as pd
import requests
import sqlite3

PWD = os.path.dirname(__file__)
CHICAGO_STOREFRONTS_DB = os.path.join(PWD, "chicago_storefronts_db.sqlite3")
CHICAGO_LICENSES_API = "https://data.cityofchicago.org/resource/xqx5-8hwx.json"
CHICAGO_LICENSES_COLUMNS = (
    "account_number,"
    "site_number,"
    "license_code,"
    "date_issued,"
    "expiration_date,"
    "latitude,"
    "longitude"
)


class Storefronts:

    def __init__(self):

        self.dbname = "Chicago Storefronts"


    def open(self):

        self.connection = sqlite3.connect(CHICAGO_STOREFRONTS_DB)
        self.cursor = self.connection.cursor()
    

    def close(self):

        assert self.connection
        self.connection.close()
    

    def create_licences_table(self):

        create_licences_table = '''
        DROP TABLE IF EXISTS licenses;
        CREATE TABLE licenses (
            account_number      INTEGER,
            site_number         INTEGER,
            license_code        VARCHAR,
            issue_date          TIMESTAMP,
            expiry_date         TIMESTAMP,
            latitude            FLOAT,
            longitude           FLOAT
        );
        '''
        self.cursor.executescript(create_licences_table)
        self.connection.commit()
    

    def populate_licenses_table(self):

        record_count = 0
        record_queue = 1000
        while record_count % record_queue == 0:
            request = requests.get(
                CHICAGO_LICENSES_API,
                params={
                    "$select": CHICAGO_LICENSES_COLUMNS,
                    "$offset": record_count,
                    "$limit": record_queue
                }
            )
            new_licenses = request.json()
            record_count += len(new_licenses)
            for new_license in new_licenses:
                new_values = (
                    new_license.get("account_number"),
                    new_license.get("site_number"),
                    new_license.get("license_code"),
                    new_license.get("date_issued"),
                    new_license.get("expiration_date"),
                    new_license.get("latitude"),
                    new_license.get("longitude")
                )
                populate_licenses_table = '''
                INSERT INTO licenses VALUES (?, ?, ?, ?, ?, ?, ?);
                '''
                self.cursor.execute(populate_licenses_table, new_values)
            self.connection.commit()


    def create_storefronts_table(self, l_bound, u_bound):

        '''
        Create a table of active storefronts and their active licenses for a
        period given by lower and upper bounds.

        l_bound (datetime64): inclusive lower bound for license expiry date.
        u_bound (datetime64): exclusive upper bound for licsene issue date.

        '''

        l_bound = pd.Timestamp(l_bound)
        u_bound = pd.Timestamp(u_bound)
        identify_extant_licenses = f'''
        SELECT DISTINCT license_code 
        FROM licenses 
        WHERE DATETIME(expiry_date) >= DATETIME('{l_bound}') 
        AND DATETIME(issue_date) < DATETIME('{u_bound}'); 
        '''
        self.cursor.execute(identify_extant_licenses)
        extant_licenses = [
            license_code[0]
            for license_code in self.cursor.fetchall()
        ]
        table_label = str(l_bound.year) + "_" + str(u_bound.year)
        create_storefronts_table = f'''
        DROP TABLE IF EXISTS storefronts_{table_label};
        CREATE TABLE storefronts_{table_label} (
            sf_id               VARCHAR, 
            earliest_issue      TIMESTAMP, 
            latest_issue        TIMESTAMP, 
            latitude            FLOAT, 
            longitude           FLOAT, 
        '''
        create_storefronts_table += ", \n".join([
            "'lc_" + license_code + "\' INTEGER DEFAULT 0"
            for license_code in extant_licenses
        ])
        create_storefronts_table += ");"
        self.cursor.executescript(create_storefronts_table)
        self.connection.commit()
        self.populate_storefronts_table(l_bound, u_bound, extant_licenses)


    def populate_storefronts_table(self, l_bound, u_bound, extant_licenses):

        '''
        Populate a table of active storefronts and their active licenses for a
        period given by lower and upper bounds.

        l_bound (Timestamp): inclusive lower bound for license expiry date.
        u_bound (Timestamp): exclusive upper bound for licsene issue date.
        extant_licenses (list): collection of licenses extant in this period.

        '''

        table_label = str(l_bound.year) + "_" + str(u_bound.year)
        populate_storefronts_table = f'''
        WITH 
            sf_ids_aggs AS ( 
                SELECT 
                    account_number || '-' || site_number AS sf_id, 
                    MIN(issue_date) AS earliest_issue, 
                    MAX(expiry_date) AS latest_issue 
                FROM licenses 
                GROUP BY account_number, site_number 
                WHERE DATETIME(expiry_date) >= DATETIME('{l_bound}') 
                AND DATETIME(issue_date) < DATETIME('{u_bound}') 
            ), 
            sf_ids_locs AS ( 
                SELECT 
                    account_number || '-' || site_number AS sf_id, 
                    latitude, 
                    longitude, 
                    RANK() OVER ( 
                        PARTITION BY account_number, site_number 
                        ORDER BY issue_date DESC 
                    ) AS last_location 
                FROM licenses 
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL 
                AND DATETIME(expiry_date) >= DATETIME('{l_bound}') 
                AND DATETIME(issue_date) < DATETIME('{u_bound}') 
            ), 
            sf_ids_complete AS ( 
                SELECT DISTINCT 
                    sf_id, 
                    earliest_issue, 
                    latest_issue, 
                    latitude, 
                    longitude 
                FROM sf_ids_aggs 
                JOIN sf_ids_locs USING (sf_id) 
                WHERE last_location = 1 
            ) 
        INSERT INTO storefronts_{table_label} (
            sf_id, 
            earliest_issue, 
            latest_issue, 
            latitude, 
            longitude 
        ) 
        SELECT * FROM sf_ids_complete;
        '''
        self.cursor.execute(populate_storefronts_table)
        for extant_license in extant_licenses:
            update_license = f'''
            UPDATE storefronts_{table_label} 
            SET 'lc_{extant_license}' = 1 
            WHERE sf_id IN ( 
                SELECT account_number || '-' || site_number AS sf_id 
                FROM licenses 
                WHERE license_code = {extant_license} 
                AND DATETIME(expiry_date) >= DATETIME('{l_bound}') 
                AND DATETIME(issue_date) < DATETIME('{u_bound}') 
            );
            '''
            self.cursor.executescript(update_license)
            self.connection.commit()

