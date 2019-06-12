#!/bin/bash

echo "Requesting census blocks from the Chicago Data Portal."
python3 db_build.py --blocks
echo "Requesting crime data from the Chicago Data Portal."
python3 db_build.py --crimes
echo "Requesting business license data from the Chicago Data Portal."
python3 db_build.py --licenses
echo "Requesting American Community Survey data from the US Census Bureau."
python3 db_build.py --census 2013
echo "Finished."
