from math import ceil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, pyplot as plt

from config import config


def perform_eda() -> None:
    """
    Perform exploratory data analysis (EDA) on datasets defined in the config.

    :return: No returned data
    :rtype: None
    """

    __drugs = []
    targets = []
    interactions = []
    valid_set_ratios = []
    test_set_ratios = []

    affinities = []
    means = []
    medians = []

    for dataset in list(config.regression_datasets.keys()):
        df_train = pd.read_parquet(f'data/raw/{config.task}/{dataset}/train')
        df_test = pd.read_parquet(f'data/raw/{config.task}/{dataset}/test')

        df_dataset = pd.concat([df_train, df_test])

        drug_count = len(df_dataset['compound_iso_smiles'].drop_duplicates())
        target_count = len(df_dataset['target_sequence'].drop_duplicates())
        interaction_count = len(df_dataset)
        test_set_ratio = round(len(df_test) / len(df_dataset) * 100, 3)

        __drugs.append(drug_count)
        targets.append(target_count)
        interactions.append(interaction_count)
        test_set_ratios.append(test_set_ratio)
        valid_set_ratios.append(20)

        affinities.append(df_dataset['affinity'].tolist())
        medians.append(float(np.median(affinities[-1])))
        means.append(float(np.mean(affinities[-1])))

    df = pd.DataFrame(
        {
            'drugs': __drugs,
            'targets': targets,
            'interactions': interactions,
            'valid_set_ratios (deducted from train split)': valid_set_ratios,
            'test_set_ratios': test_set_ratios,
        },
        index=list(config.regression_datasets.values()),
    )

    fig = plt.figure(figsize=(15, 10))

    num_of_columns = 3

    gs = gridspec.GridSpec(
        ceil(len(list(config.regression_datasets.keys())) / num_of_columns),
        num_of_columns,
        figure=fig,
    )

    for idx, dataset in enumerate(list(config.regression_datasets.keys())):
        y = affinities[idx]
        mean = means[idx]
        median = medians[idx]

        gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[idx], height_ratios=(0.15, 1))

        ax1 = fig.add_subplot(gs00[0])
        ax2 = fig.add_subplot(gs00[1])

        sns.boxplot(y, ax=ax1, orient='h').set_title(list(config.regression_datasets.values())[idx])

        sns.histplot(
            y, ax=ax2, bins=50, kde=True,
            stat='density', kde_kws=dict(cut=3),
            alpha=.4, edgecolor=(1, 1, 1, .4),
        )

        ax1.axvline(median, color='b', linestyle='--')
        ax1.axvline(mean, color='g', linestyle='--')
        ax1.set(xlabel='')
        ax1.set_xticks([])
        ax2.axvline(median, color='b', linestyle='--')
        ax2.axvline(mean, color='g', linestyle='--')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim([0, 7])
        ax2.legend(['KDE', 'Median', 'Mean'])

    plt.show()

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    print(df)
