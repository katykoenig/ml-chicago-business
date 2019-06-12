#final as of tue 540pm


writing results for data/valid_2014-05-31_2014-06-01.csv
BEST MODEL FOR auc_roc
model                  knn
precision_at_5    0.759353
accuracy_at_5     0.331733
f1_score_at_5     0.102018
recall_at_5       0.054682
auc_roc           0.525877
Name: 61, dtype: object
{'weights': 'distance', 'n_neighbors': 50}

0.1948815790757881 storefronts_on_block
0.01621416689127304 police_district
0.0493339331730308 pct_medium_travel_time
0.023001848984061914 pct_below_median_income
0.024100468468792536 pct_black
0.013764464216348169 pct_hispanic
0.22658502441293218 census_tract
0.08092599023941097 THEFT
0.06446931746765185 CRIMINAL TRESPASS
0.01810980388178879 MOTOR VEHICLE THEFT
0.024036446414370694 Total_Arrests
0.2570895624754538 month_issue





writing results for data/valid_2015-05-31_2015-06-01.csv
BEST MODEL FOR auc_roc
model                   knn
precision_at_5     0.770826
accuracy_at_5      0.330156
f1_score_at_5      0.103184
recall_at_5       0.0552929
auc_roc            0.522634
Name: 61, dtype: object
{'weights': 'distance', 'n_neighbors': 50}


0.22747386243636206 storefronts_on_block
0.03559735800655597 police_district
0.12864758980404808 block_group
0.03503325647064587 pct_male_children
0.03678334838629126 pct_high_travel_time
0.026555257934462215 pct_below_poverty
0.025476858499809883 pct_below_median_income
0.03993770484769659 pct_black
0.09548955537895247 pct_hispanic
0.016028605494256654 OTHER OFFENSE
0.014612866864426229 THEFT
0.026577190071570975 BURGLARY
0.03886961829267243 KIDNAPPING
0.2398301817576395 month_issue


writing results for data/valid_2016-05-31_2016-06-01.csv
BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.700637
accuracy_at_5          0.350147
f1_score_at_5         0.0973071
recall_at_5           0.0522843
auc_roc                0.526554
Name: 18, dtype: object
{'n_jobs': -1, 'max_depth': 1, 'n_estimators': 10, 'min_samples_split': 2}

0.2033889245880266 storefronts_on_block
0.035891309097612005 police_district
0.04786270494904698 block_group
0.03297597532288257 pct_male_working
0.0693185766692661 pct_male_elderly
0.03312503569038093 pct_high_travel_time
0.011494951900626918 pct_below_poverty
0.09440812237703534 pct_above_median_income
0.04962150306379914 pct_black
0.06305882797512172 pct_asian
0.033340172732757696 pct_hispanic
0.017600702717251848 census_tract
0.044082552785588786 THEFT
0.018457222658087673 GAMBLING
0.23841540885273052 month_issue


writing results for data/valid_2017-05-31_2017-06-01.csv
BEST MODEL FOR auc_roc
model             random_forest
precision_at_5         0.625338
accuracy_at_5           0.41287
f1_score_at_5         0.0962287
recall_at_5           0.0521249
auc_roc                0.528473
Name: 26, dtype: object
{'n_jobs': -1, 'max_depth': 1, 'n_estimators': 10, 'min_samples_split': 10}


0.1547413330639465 storefronts_on_block
0.0406427375423391 police_district
0.01507980953859365 block_group
0.08596668754818339 total_bachelors_degrees
0.03145075498338729 total_population
0.04211129963741711 pct_male_working
0.0424295487988854 pct_female_working
0.05185265668933306 pct_low_travel_time
0.038303627746717785 pct_below_median_income
0.031134557170196427 pct_white
0.013172990183963201 census_tract
0.03728184580067102 THEFT
0.01590966734217681 NARCOTICS
0.022701649864633737 CRIMINAL TRESPASS
0.08637195735852941 Total_Arrests
0.020746704947906345 Total_Domestic
0.27010217178311974 month_issue




features
['storefronts_on_block',
 'police_district',
 'block_group',
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
Params: {'n_jobs': -1, 'min_samples_split': 2, 'n_estimators': 100, 'max_depth': 1}




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



# without license type

model                                                 random_forest
parameters        {'max_depth': 5, 'n_jobs': -1, 'min_samples_sp...
precision_at_5                                             0.540749
accuracy_at_5                                              0.404413
f1_score_at_5                                              0.083212
recall_at_5                                               0.0450741
auc_roc                                                    0.508558
Name: 41, dtype: object
0.15479645439242826 storefronts_on_block
0.04065721513305201 police_district
0.08599731025207903 total_bachelors_degrees
0.03146195824344789 total_population
0.02803161918674767 pct_male_working
0.04244466288023889 pct_female_working
0.051871127431864686 pct_low_travel_time
0.03831727210925018 pct_below_median_income
0.03114564779555768 pct_white
0.027906647886023463 census_tract
0.03729512618819318 THEFT
0.015915334619188322 NARCOTICS
0.02270973655404302 CRIMINAL TRESPASS
0.034848776421386896 OFFENSE INVOLVING CHILDREN
0.08640272442598905 Total_Arrests
0.2701983864805099 month_issue
{'max_depth': 5, 'n_jobs': -1, 'min_samples_split': 5, 'n_estimators': 2000}




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


