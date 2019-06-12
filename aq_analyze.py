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
import argparse
import pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot

METRICS = [
    "for_disparity",    # false omission rate
    "pprev_disparity"   # predicted prevalence
]


def aq_analyze(arguments):

    # Import valid set and block scores from best model.
    block_scores = pd.read_csv(arguments.block_scores) \
        .rename(columns={0: "block_group", 1: "score"})
    valid_set = pd.read_csv(arguments.valid_set) \
        .drop_duplicates(subset="block_group") \
        .merge(block_scores, on="block_group") \
        .rename(columns={"success": "label_value"})
    features_to_drop = ["block", "block_group"] + [
        column
        for column in valid_set.columns
        if ("crime" in column or "license" in column) and "count" not in column
    ]
    valid_set = valid_set.drop(columns=features_to_drop)
    # Preprocess outside of Aequitas because preprocess_input_df() doesn't work.
    for column in valid_set.columns:
        if column == "score":
            valid_set[column] = valid_set[column].astype(float)
        elif column == "label_value":
            valid_set[column] = valid_set[column].astype(int)
        else:
            if valid_set[column].nunique() > 1:
                valid_set[column] = pd.qcut(
                    x=valid_set[column],
                    q=4,
                    precision=2,
                    duplicates="drop"
                )
            valid_set[column] = valid_set[column].astype(str)
    # Initialize Aequitas objects and export directory.
    aqg, aqb, aqf, aqp = Group(), Bias(), Fairness(), Plot()
    directory = "aequitas"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Calculate crosstabs by distinct group.
    crosstabs, _ = aqg.get_crosstabs(
        df=valid_set,
        score_thresholds={"score": [arguments.threshold]}
    )
    absolute_metrics = aqg.list_absolute_metrics(crosstabs)
    crosstabs[["attribute_name", "attribute_value"] + absolute_metrics] \
        .round(2) \
        .to_csv(directory + "/aequitas_crosstabs.csv", index=False)
    # Calculate and plot bias with respect to white, high income communities.
    base_comparison = {
        "pct_white": "(0.67, 1.0]",
        "pct_high_income": "(0.25, 0.73]"
    }
    disparity_white_hiinc = aqb.get_disparity_predefined_groups(
        crosstabs.loc[crosstabs["attribute_name"].isin(base_comparison.keys())],
        valid_set, base_comparison
    )
    a = aqp.plot_disparity_all(disparity_white_hiinc, metrics=METRICS)
    a.save_fig(directory + "/bias_white_high_income.png")
    # Calculate and plot fairness with respect to the same.
    b = aqp.plot_fairness_disparity_all(
        aqf.get_group_value_fairness(disparity_white_hiinc),
        metrics=METRICS
    )
    b.save_fig(directory + "/fairness_white_high_income.png")
    # Calculate and plot bias with respect to majority group.
    disparity_major_group = aqb.get_disparity_major_group(
        crosstabs, valid_set
    )
    c = aqp.plot_disparity_all(disparity_major_group, metrics=METRICS)
    c.save_fig(directory + "/bias_majority_group.png")
    # Calculate and plot fairness with respect to the same.
    d = aqp.plot_fairness_disparity_all(
        aqf.get_group_value_fairness(disparity_major_group),
        metrics=METRICS
    )
    d.save_fig(directory + "/fairness_majority_group.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Aequitas on the Chicago Entrepreneurship data."
    )
    parser.add_argument(
        "--block_scores",
        help="Path to the CSV file of classifier scores for each block group.",
        dest="block_scores"
    )
    parser.add_argument(
        "--valid_set",
        help="Path to CSV file of the validation set.",
        dest="valid_set"
    )
    parser.add_argument(
        "--threshold",
        default=0.8,
        help="Threshold at which to assign scores to the positive class.",
        dest="threshold"
    )
    arguments = parser.parse_args()
    aq_analyze(arguments)
