"""
This script serves two purposes:
1. It plots the emotion distribution for fake and real news headlines and saves the plot to the figs folder.
2. It prints a table with the emotion distribution for fake and real news headlines.

Author: Laura Bock Paulsen (202005791@post.au.dk)
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'serif'

def most_prevalent(data:pd.DataFrame):
    """
    Returns a dataframe with the most prevalent emotion label for each headline.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with real and fake news headlines and score for each emotion.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with added column with the most prevalent emotion label for each headline.
    """
    only_numeric = data.select_dtypes(include=np.number)

    data['emotion_label'] = only_numeric.apply(lambda x: only_numeric.columns[x.argmax()], axis = 1)

    return data
    

def plot_emotion_counts(data:pd.DataFrame, save_path:Path = None):
    """
    Plots a stacked bar plot with the emotion distribution for fake and real news headlines. 

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with real and fake news headlines and their emotion labels.

    save_path: Path, optional
        Path to save the plot to. If None, the plot is not saved.
    
    Returns
    -------
    fig, ax: matplotlib.pyplot.figure, matplotlib.pyplot.axes
        Figure and axes of the plot.
    """
    fig, ax = plt.subplots(1, dpi = 300, figsize = (10, 7))
    
    # counts of emotions grouped by label
    emo_counts = pd.crosstab(data["emotion_label"], data["label"])

    # sort by count
    emo_counts = emo_counts.sort_values(by = "FAKE", ascending = False)

    # list of emotions to use as labels
    emotions = list(emo_counts.index)
    emotions = [emo.capitalize() for emo in emotions]

    # keeping track of bottom of each bar (for stacked bar plot)
    bottom = np.zeros(7)

    # plotting stacked bar plot
    width = 0.7

    palette = sns.color_palette("Spectral", 7)

    for label, count in zip(["Fake news", "Real news"], [emo_counts["FAKE"], emo_counts["REAL"]]):
        # bottom bar
        if bottom.sum() == 0: 
            p = ax.bar(emotions, count, width, label=label, bottom=bottom, color = palette[0], alpha = 0.9)
            bottom += count
        
        # top bar 
        else: 
            p = ax.bar(emotions, count, width, label=label, bottom=bottom, color = palette[1], alpha = 0.9)

        ax.bar_label(p, label_type='center', color = "black")

    ax.legend()
    
    # rotate xticks
    ax.tick_params(axis='x', which='major', labelsize=14, rotation = 45)

    # set label and title
    ax.set_ylabel("Count", fontsize = 14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

def plot_proportion_emotions(data:pd.DataFrame, save_path:Path = None):
    """
    Plots the proportion of emotions for fake and real news headlines.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with real and fake news headlines and their emotion labels.

    save_path: Path, optional
        Path to save the plot to. If None, the plot is not saved.
    
    Returns
    -------
    fig, ax: matplotlib.pyplot.figure, matplotlib.pyplot.axes
        Figure and axes of the plot.
    """
    fig, ax = plt.subplots(1, dpi = 300, figsize = (10, 7))
    
    # counts of emotions grouped by label
    emo_counts = pd.crosstab(data["emotion_label"], data["label"])

    # change to proportions
    emo_counts = emo_counts.div(emo_counts.sum(axis=0), axis=1).mul(100)

    # list of emotions to use as labels
    emotions = list(emo_counts.index)
    emotions = [emo.capitalize() for emo in emotions]

    # keeping track of bottom of each bar (for stacked bar plot)
    bottom = np.zeros(2)

    # plotting stacked bar plot
    width = 0.9
    
    # colours
    palette = sns.color_palette("Spectral", 7)
    
    # loop over emotions
    for i, (emo, count) in enumerate(zip(emotions, emo_counts.values)):
        p = ax.bar(["Fake news", "Real news"], count, width, label=emo, bottom=bottom, color = palette[i], alpha = 0.9)
        bottom += count

        # label with rounded percentage
        ax.bar_label(p, labels = [str(round(x, 2)) + "%" for x in count], color = "black", label_type='center', fontsize = 14)

    ax.legend(loc = "upper center", ncol = 7, fontsize = 14, bbox_to_anchor=(0.5, 1.05))
    
    # remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    # remove ticks
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_yticklabels([])

    # increase font size x 
    ax.tick_params(axis='x', which='major', labelsize=14)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

def print_table(data):
    """
    Prints a table formatted in markdown with the proportion of emotions for fake and real news headlines as a percentage.
    
    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with real and fake news headlines and their emotion labels.

    Returns
    -------
    None
    """
    count_table = pd.crosstab(data["label"], data["emotion_label"])

    # add row with total
    count_table.loc["Total"] = count_table.sum()

    # percentage
    count_table = count_table.div(count_table.sum(axis=1), axis=0).mul(100)

    # round to 2 decimals
    count_table = count_table.round(2)

    print(count_table.to_markdown())


def main():
    path = Path(__file__)
    in_path = path.parents[1] / "data" / "classified_emotions.csv"
    fig_path = path.parents[1] / "figs" 
    
    data = pd.read_csv(in_path)
    data = most_prevalent(data)

    # plot
    fig, ax = plot_emotion_counts(data, save_path = fig_path / "emotion_counts.png")

    # proportion plot
    fig, ax = plot_proportion_emotions(data, save_path = fig_path / "emotion_proportions.png")

    # print table
    print_table(data)


if __name__ == '__main__':
    main()
