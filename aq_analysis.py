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
DROP_COLUMN_KEYWORDS = [
    "crime", "license", "lbound", "ubound", "block", "block_group"
]


def aq_analysis(arguments):

    # Import result set from best model.
    result_set = pd.read_csv(arguments.result_set) \
        .drop_duplicates(subset="block_group") \
        .rename(columns={"successful": "label_value"})
    # Drop columns not needed for analysis.
    features_to_drop = [
        column
        for column in result_set.columns
        if column in DROP_COLUMN_KEYWORDS and "count" not in column
    ]
    result_set = result_set.drop(columns=features_to_drop)
    # Initialize base comparison attributes dictionary.
    base_comparison = {
        "pct_white": None,
        "pct_high_income": None
    }
    base_comparison_label = "_".join(base_comparison.keys())
    # Preprocess outside of Aequitas because preprocess_input_df() doesn't work.
    for column in result_set.columns:
        if column == "score":
            result_set[column] = result_set[column].astype(float)
        elif column == "label_value":
            result_set[column] = result_set[column].astype(int)
        else:
            if result_set[column].nunique() > 1:
                result_set[column], bins = pd.qcut(
                    x=result_set[column],
                    q=4,
                    precision=2,
                    duplicates="drop",
                    retbins=True
                )
                # Save label of highest quartile for base comparison attributes.
                if column in base_comparison:
                    lb = str(round(bins[3], 2))
                    ub = str(round(bins[4], 2))
                    base_comparison[column] = "(" + lb + ", " + ub + "]"
            result_set[column] = result_set[column].astype(str)
    # Initialize Aequitas objects and export directory.
    aqg, aqb, aqf, aqp = Group(), Bias(), Fairness(), Plot()
    directory = "aequitas"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Calculate crosstabs by distinct group.
    crosstabs, _ = aqg.get_crosstabs(
        df=result_set,
        score_thresholds={"score": [float(arguments.threshold)]}
    )
    absolute_metrics = aqg.list_absolute_metrics(crosstabs)
    crosstabs[["attribute_name", "attribute_value"] + absolute_metrics] \
        .round(2) \
        .to_csv(directory + "/aequitas_crosstabs.csv", index=False)
    # Plot bias and fairness with respect to white, high income communities.
    disparity_white_hiinc = aqb.get_disparity_predefined_groups(
        crosstabs.loc[crosstabs["attribute_name"].isin(base_comparison.keys())],
        result_set, base_comparison
    )
    a = aqp.plot_disparity_all(
        disparity_white_hiinc,
        metrics=METRICS,
        show_figure=False
    )
    a_filename = "bias_ref_" + base_comparison_label + ".png"
    a.savefig(directory + "/" + a_filename)
    b = aqp.plot_fairness_disparity_all(
        aqf.get_group_value_fairness(disparity_white_hiinc),
        metrics=METRICS,
        show_figure=False
    )
    b_filename = "fairness_ref_" + base_comparison_label + ".png"
    b.savefig(directory + "/" + b_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Aequitas on the Chicago Entrepreneurship data."
    )
    parser.add_argument(
        "--result_set",
        help="Path to CSV file of the validation set.",
        dest="result_set"
    )
    parser.add_argument(
        "--threshold",
        default=0.8207,
        help="Threshold at which to assign scores to the positive class.",
        dest="threshold"
    )
    arguments = parser.parse_args()
    aq_analysis(arguments)
