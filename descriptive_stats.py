'''
Katy Koenig
CAPP 30254


Functions for Getting Descriptive & Data Exploration Stats
'''
import matplotlib.pyplot as plt
import plotnine as p9
import seaborn as sns


def summary_stats(dataframe):
    '''
    Creates table of summary statistics for each column of dataframe

    Input: a pandas dataframe

    Output: a table
    '''
    summary = dataframe.describe()
    return summary


def evaluate_correlations(dataframe, output_filename):
    '''
    Presents information regarding the correlation of each variable/column of a dataframe

    Input: a pandas dataframe

    Outputs:
        corr_df: a dataframe, showing the correlation between each variable
        corr_heatmap: a heatmap, reflecting the correlation between each variable
    '''
    corr_df = dataframe.corr()
    corr_heatmap = sns.heatmap(corr_df, xticklabels=corr_df.columns, yticklabels=corr_df.columns)
    plt.title("Correlation of Variables")
    corr_heatmap.figure.savefig(output_filename, bbox_inches="tight")
    plt.clf()
    return corr_df


def show_distribution(dataframe):
    '''
    Saves a histogram for each column of the dataframe

    Inputs: a pandas dataframe

    Outputs: None
    '''
    dataframe.hist(grid=False, sharey=True, alpha=0.5, figsize=(20, 10))
    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.clf()


def create_scatterplots(dataframe, unique_id='unique_id'):
    '''
    Creates and saves scatterplots for each column in a dataframe

    Inputs:
        dataframe: a pandas dataframe
        unique_id: a pandas series representing a unique identifier for each observation

    Outputs: None
    '''
    reset_df = dataframe.reset_index()
    for column in dataframe.columns:
        file_name = str(column) + 'scatterplot' + '.png'
        plt1 = p9.ggplot(reset_df, p9.aes(x=column, y=unique_id)) + p9.geom_point()
        print('Saving scatterplot: '  + file_name)
        p9.ggsave(filename=file_name, plot=plt1, device='png')


def check_null_values(dataframe):
    '''
    Counts the number of null values in each column of a dataframe

    Input: a pandas dataframe

    Output: a pandas series with the number of null values in each column
    '''
    return dataframe.isnull().sum(axis=0)
