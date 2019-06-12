# Promoting Sustained Entrepreneurship in Chicago
Machine Learning for Public Policy
University of Chicago, CS & Harris School of Public Policy

## Collaborators
* Dr. Rayid Ghani (@rayidghani)
* Katy Koenig (@katykoenig)
* Eric Langowski (@erhla)
* Patrick Lavallee Delgado (@lavalleedelgado)

## Introduction

Chicago is “the city that works.” In January 2018, the Illinois Department of Employment Security announced that almost 175,000 jobs had been created in Chicago since 2010. This growth is not indicative of a healthy region, however, as the majority of new jobs were in Chicago’s Loop and the Chicago area population has consistently declined over the same time period. Creating healthy businesses in every community is critical for the city’s future sustainable growth. An unhealthy business community can lead to negative outcomes for neighborhoods and negative externalities for the city, as declining tax bases force resource to be cut in death spirals. Or as Sociologist Robert J. Smapson wrote in Great American City, “bad locations mean bad business.”

Alongside long standing narratives of segregation and racial inequity, the current business environment is indicative of structural and systematic under- and disinvestment in Chicago’s communities of color. People of color own only 32% of businesses in Chicago while only 32% of Chicago’s population identify as non-Hispanic white. The average white-owned business is valued more than 12x Black-owned business in Chicago. We aim to offer actionable analysis and policy recommendations that equip the City to improve the state of entrepreneurship and position traditionally underinvested communities towards sustainable business growth.

We seek to obtain the attributes of a successful business for use in subsidizing the initial capital investment of new businesses throughout Chicago. Through analyzing the businesses for which our models deem the most successful, we are able to find the attributes which indicate whether a new business will be successful. In application, we will use these important features to justify the granting of subsidies to new businesses by the City of Chicago.

## Getting started

This repository includes the machine learning pipeline and analysis for this study.

### Requirements

* python 3.7.3
* numpy 1.16.2
* pandas 0.24.2
* geopandas 0.4.1
* scikit-learn 0.21.2
* matplotlib 3.0.3
* any dependencies

### Running our code

We've included two shell scripts to execute our data collection and warehousing:
* request_data_from_source.sh
* request_train_valid_sets.sh

Alternatively, you may rebuild the database piecemeal from the command line:
```
$ python3 db_build.py --blocks
$ python3 db_build.py --licenses
$ python3 db_build.py --crimes
$ python3 db_build.py --census 2013
```

And you may compile your own testing and validation sets from the database:
```
$ python3 db_query.py --train_lb 2010-01-01 --train_ub 2015-06-01 --valid_lb 2017-06-01 --valid_ub 2017-06-02
```

With the data in hand, you may rerun our analysis from the command line:
```
$ python3 ml_pipeline.py
$ python3 aq_analysis.py --result_set results_valid_2017-06-01_2017-06-02.csv
```

We recommend you limit the load you assign to `ml_pipeline.py` by passing a model name through the `--model` flag. Please note that to model on any temporal set other than those our analysis considers, you must modify the `TEMPORAL_SPLITS` constant in `ml_pipeline.py` with the correct lower and upper bound dates so that the module may locate the appropriate CSV files.

Finally, please find our analysis in this repository. 

