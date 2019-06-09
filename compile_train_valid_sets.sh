#!/bin/bash

echo "Compiling training set from 2010 through 2015, validation set for 2017."
python3 db_query.py --train_lb 2010-01-01 --train_ub 2015-06-01 --valid_lb 2017-05-31 --valid_ub 2017-06-01
echo "Compiling training set from 2010 through 2014, validation set for 2016."
python3 db_query.py --train_lb 2010-01-01 --train_ub 2014-06-01 --valid_lb 2016-05-31 --valid_ub 2016-06-01
echo "Compiling training set from 2010 through 2013, validation set for 2015."
python3 db_query.py --train_lb 2010-01-01 --train_ub 2013-06-01 --valid_lb 2015-05-31 --valid_ub 2015-06-01
echo "Compiling training set from 2010 through 2012, validation set for 2014."
python3 db_query.py --train_lb 2010-01-01 --train_ub 2012-06-01 --valid_lb 2014-05-31 --valid_ub 2014-06-01
echo "Finished."
