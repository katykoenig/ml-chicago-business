'''
Plumbum: a machine learning pipeline.
Plotting module

Patrick Lavallee Delgado
University of Chicago, CS & Harris MSCAPP '20
May 2019

'''

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def _plot_curves(curves, title, x_axis, y_axis):

    # Plot curves.
    color_map_scalar = max(256 // len(curves), 1)
    for i, curve in enumerate(curves):
        plt.plot(
            curve.index,
            curve,
            color=plt.cm.magma(color_map_scalar * i),
            alpha=0.6,
            linewidth=2,
            label=curve.name
        )
    # Set labels and plot attributes.
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.legend(
        bbox_to_anchor=(1.04,0.5),
        loc="center left",
        borderaxespad=0
    )
    # Save plot.
    filename = "_".join(title.lower().split()) + ".png"
    plt.savefig(
        fname=filename,
        bbox_inches="tight"
    )
    plt.clf()


def _plot_distributions(data):

    title = "Variable Distributions"
    # Calculate plot grid dimensions.
    grid_cols = 3
    grid_rows = np.ceil(len(data.columns) / grid_cols)
    plt.figure()
    # Plot histograms or bar charts of each variable.
    for i, variable in enumerate(data.columns, 1):
        plt.subplot(grid_rows, grid_cols, i)
        # Drop any null observations to avert histogram errors.
        variable = data[variable].loc[data[variable].notnull()]
        if pd.api.types.is_string_dtype(variable):
            categorical = variable.value_counts()
            plt.bar(
                x=np.array(categorical.index),
                height=np.array(categorical.values),
                alpha=0.6
            )
        else:
            plt.hist(
                x=variable,
                bins=50,
                alpha=0.6
            )
        plt.grid(True)
        # Add title to plot at top center.
        if i == 2:
            plt.title(title)
        # Add axis labels.
        plt.ylabel("Frequency")
        plt.xlabel(" ".join(variable.split("_")).title())
    # Save plot grid.
    filename = "_".join(title.lower().split()) + ".png"
    plt.savefig(
        fname=filename,
        bbox_inches="tight"
    )
    plt.clf()


def _plot_correlations(data):

    title = "Variable Correlations"
    fig, ax = plt.subplots()
    # Plot heatmap.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )
    # Set labels and plot attributes.
    im = ax.imshow(data, cmap="magma")
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(list(data.index))
    ax.set_yticklabels(list(data.columns))
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    # Save plot.
    filename = "_".join(title.lower().split()) + ".png"
    plt.savefig(
        fname=filename
    )
    plt.clf()
