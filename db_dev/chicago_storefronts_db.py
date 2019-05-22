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

import pandas as pd
import requests
import psycopg2 as pg

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

    def __init__(self, name="Chicago Storefronts"):

        self.dbhost = "127.0.0.1"
        self.dbname = name


    def open(self):

        credentials = "host=" + self.dbhost + " dbname=" + self.dbname
        self.connection = pg.connect(credentials)
    

    def close(self):

        assert self.connection and not self.connection.closed
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
            latitude            NUMERIC,
            longitude           NUMERIC
        );
        '''
        cursor = self.connection.cursor()
        cursor.execute(create_licences_table)
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
            new_license_data = request.json()
            record_count += len(new_license_data)
            populate_licenses_table = '''
            COPY licenses FROM STDIN WITH FORCE NULL;
            '''
            cursor = self.connection.cursor()
            cursor.copy_expert(populate_licenses_table, new_license_data)
            self.connection.commit()


    def create_storefronts_table(self):

        create_storefronts_table = '''
        DROP TABLE IF EXISTS storefronts;
        CREATE TABLE storefronts (
            sf_id               VARCHAR,
            earliest_issue      TIMESTAMP,
            latest_issue        TIMESTAMP,
            latitude            NUMERIC,
            longitude           NUMERIC
        );
        '''
        cursor = self.connection.cursor()
        cursor.execute(create_storefronts_table)
        self.connection.commit()


    def populate_storefronts_table(self):

        populate_storefronts_table = '''
        WITH
            sf_ids_aggs AS (
                SELECT 
                    account_number,
                    site_number, 
                    MIN(issue_date) AS earliest_issue, 
                    MAX(expiry_date) AS latest_issue
                FROM licenses 
                GROUP BY account_number, site_number
            ), 
            sf_ids_locs AS (
                SELECT
                    account_number, 
                    site_number, 
                    latitude, 
                    longitude, 
                    RANK() OVER (
                        PARTITION BY account_number, site_number 
                        ORDER BY issue_date DESC
                    ) AS last_location 
                FROM licenses 
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL 
            ), 
            sf_ids_complete AS (
                SELECT 
                    FORMAT (account_number || '-' || site_number) AS sf_id, 
                    earliest_issue, 
                    latest_issue, 
                    latitude, 
                    longitude 
                FROM sf_ids_aggs 
                JOIN sf_ids_locs USING (account_number, site_number) 
                WHERE last_location = 1 
            ) 
        INSERT INTO storefronts SELECT * FROM sf_ids_complete;
        '''
        cursor = self.connection.cursor()
        cursor.execute(populate_storefronts_table)
        self.connection.commit()

