#with census tracts (see feature lst below)

writing results for data/valid_2014-05-31_2014-06-01.csv
BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.743525
accuracy_at_5          0.330151
f1_score_at_5         0.0998913
recall_at_5           0.0535423
auc_roc                0.507654
Name: 35, dtype: object
0.12835714820510474 license_1329
0.5824916621174718 license_1010
0.03603147122741269 license_1012
0.24516223309230747 license_1006

writing results for data/valid_2015-05-31_2015-06-01.csv
BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.854128
accuracy_at_5          0.338485
f1_score_at_5          0.114335
recall_at_5           0.0612684
auc_roc                0.513775
Name: 19, dtype: object
0.42306505426414875 license_1010
0.2177845639825049 license_1006
0.21820460982031722 license_4404
0.07222295957877867 license_4406
0.06835863058581161 license_8340

writing results for data/valid_2016-05-31_2016-06-01.csv
BEST MODEL FOR auc_roc
model                   knn
precision_at_5     0.678531
accuracy_at_5      0.347937
f1_score_at_5      0.094237
recall_at_5       0.0506347
auc_roc            0.511265
Name: 69, dtype: object
0.4041886014992585 license_1010
0.21130942171627096 license_1006
0.23021155342151098 license_4404
0.0850307322665601 license_4406
0.06851847243109083 license_8340

BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.652375
accuracy_at_5          0.415573
f1_score_at_5          0.100389
recall_at_5           0.0543786
auc_roc                 0.50921
Name: 19, dtype: object
0.6176142104505442 license_1010
0.010020747651007265 license_1011
0.09182518166756577 license_1012
0.12521408731145942 license_1006
0.1549213150622894 license_4404




#with patrick's crime cols and block group

writing results for data/valid_2014-05-31_2014-06-01.csv
BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.732374
accuracy_at_5          0.329036
f1_score_at_5         0.0983931
recall_at_5           0.0527393
auc_roc                0.508311
Name: 47, dtype: object
0.1283492891159242 license_1329
0.5824559971468526 license_1010
0.036029265081906625 license_1012
0.24514722222707308 license_1006


writing results for data/valid_2015-05-31_2015-06-01.csv
BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.696039
accuracy_at_5          0.322679
f1_score_at_5         0.0931731
recall_at_5           0.0499283
auc_roc                0.517574
Name: 21, dtype: object
0.423035494502728 license_1010
0.21776934726893435 license_1006
0.21801351110953732 license_4404
0.07221791332541236 license_4406
0.0683538543349944 license_8340


writing results for data/valid_2016-05-31_2016-06-01.csv
BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.585613
accuracy_at_5          0.338647
f1_score_at_5         0.0813321
recall_at_5           0.0437007
auc_roc                0.511728
Name: 46, dtype: object
0.40418468430461235 license_1010
0.21130737381058445 license_1006
0.23020932232577693 license_4404
0.08502990819104694 license_4406
0.06851780838417715 license_8340

writing results for data/valid_2017-05-31_2017-06-01.csv
BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.491696
accuracy_at_5           0.39951
f1_score_at_5         0.0756635
recall_at_5           0.0409852
auc_roc                0.508102
Name: 34, dtype: object
0.6176142104505442 license_1010
0.010020747651007265 license_1011
0.09182518166756577 license_1012
0.12521408731145942 license_1006
0.1549213150622894 license_4404


#features
['storefronts_on_block',
 'license_1329',
 'license_1470',
 'license_1002',
 'license_1481',
 'license_1474',
 'license_2101',
 'license_1480',
 'license_8343',
 'license_1056',
 'license_1005',
 'license_1782',
 'license_1473',
 'license_1683',
 'license_1568',
 'license_1571',
 'license_8344',
 'license_1375',
 'license_1003',
 'license_1016',
 'license_1070',
 'license_1690',
 'license_1064',
 'license_1479',
 'license_1013',
 'license_1587',
 'license_1255',
 'license_1374',
 'license_1004',
 'license_1030',
 'license_1572',
 'license_1594',
 'license_1685',
 'license_1900',
 'license_1372',
 'license_8345',
 'license_1608',
 'license_1055',
 'license_1609',
 'license_1505',
 'license_1682',
 'license_1833',
 'license_1330',
 'license_1573',
 'license_1478',
 'license_1483',
 'license_1080',
 'license_1550',
 'license_1688',
 'license_8342',
 'license_3717',
 'license_1034',
 'license_1686',
 'license_1010',
 'license_1011',
 'license_1020',
 'license_1603',
 'license_1605',
 'license_1012',
 'license_1607',
 'license_1606',
 'license_1781',
 'license_1006',
 'license_1133',
 'license_1476',
 'license_1477',
 'license_1275',
 'license_1475',
 'license_1050',
 'license_1371',
 'license_1008',
 'license_1604',
 'license_1009',
 'license_1054',
 'license_1586',
 'license_1370',
 'license_1058',
 'license_1676',
 'license_1046',
 'license_1060',
 'license_1316',
 'license_1842',
 'license_1786',
 'license_1062',
 'license_1800',
 'license_1569',
 'license_1931',
 'license_1930',
 'license_1784',
 'license_1932',
 'license_1840',
 'license_1841',
 'license_1039',
 'license_1061',
 'license_1315',
 'license_1431',
 'license_8100',
 'license_1253',
 'license_1625',
 'license_1684',
 'license_1007',
 'license_1482',
 'license_1570',
 'license_1057',
 'license_1456',
 'license_1033',
 'license_1053',
 'license_1524',
 'license_1585',
 'license_1584',
 'license_1014',
 'license_1472',
 'license_1783',
 'license_1471',
 'police_district',
 'total_bachelors_degrees',
 'total_population',
 'pct_male_children',
 'pct_male_working',
 'pct_male_elderly',
 'pct_female_children',
 'pct_female_working',
 'pct_female_elderly',
 'pct_low_travel_time',
 'pct_medium_travel_time',
 'pct_high_travel_time',
 'pct_below_poverty',
 'pct_below_median_income',
 'pct_above_median_income',
 'pct_high_income',
 'pct_white',
 'pct_black',
 'pct_asian',
 'pct_hispanic',
 'census_tract',
 'HOMICIDE',
 'OTHER OFFENSE',
 'ROBBERY',
 'THEFT',
 'NARCOTICS',
 'BATTERY',
 'ASSAULT',
 'CRIMINAL DAMAGE',
 'CRIMINAL TRESPASS',
 'PUBLIC PEACE VIOLATION',
 'MOTOR VEHICLE THEFT',
 'DECEPTIVE PRACTICE',
 'WEAPONS VIOLATION',
 'INTERFERENCE WITH PUBLIC OFFICER',
 'BURGLARY',
 'CRIM SEXUAL ASSAULT',
 'OFFENSE INVOLVING CHILDREN',
 'PUBLIC INDECENCY',
 'SEX OFFENSE',
 'KIDNAPPING',
 'PROSTITUTION',
 'INTIMIDATION',
 'ARSON',
 'LIQUOR LAW VIOLATION',
 'CONCEALED CARRY LICENSE VIOLATION',
 'GAMBLING',
 'OTHER NARCOTIC VIOLATION',
 'STALKING',
 'OBSCENITY',
 'HUMAN TRAFFICKING',
 'NON-CRIMINAL',
 'NON-CRIMINAL (SUBJECT SPECIFIED)',
 'NON - CRIMINAL',
 'Total_Crimes',
 'Total_Arrests',
 'Total_Domestic',
 'month_issue']


