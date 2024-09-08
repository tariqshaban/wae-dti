import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from config import config


def __plot_results_cls(actual: np.ndarray, expected: np.ndarray, path: str = None) -> None:
    """
    Generates and saves an AUC and AUPRC plots for classification predictions.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values
    :param str path: Location where to save the generated plot

    :return: No returned data
    :rtype: None
    """

    plt.figure(figsize=(7, 6))

    fpr, tpr, _ = roc_curve(actual, expected)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, tpr, color='b', lw=1.5,
        label=f'AUC={roc_auc:.2f}'
        if path else
        f'{config.datasets[config.dataset]} (AUC={roc_auc:.2f})',
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Receiver Operating Characteristic Curve', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=18)

    plt.tight_layout()

    if path:
        plt.savefig(f'{path}/auc.png')
    else:
        plt.savefig(
            f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/auc.png')

    plt.clf()
    plt.figure(figsize=(7, 6))

    precision, recall, _ = precision_recall_curve(actual, expected)
    average_precision = average_precision_score(actual, expected)
    plt.plot(
        recall, precision, color='b', lw=1.5,
        label=f'AP={average_precision:.2f}'
        if path else
        f'{config.datasets[config.dataset]} (AP={average_precision:.2f})',
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('Precision-Recall Curve', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower left', fontsize=18)

    plt.tight_layout()

    if path:
        plt.savefig(f'{path}/auprc.png')
    else:
        plt.savefig(
            f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/auprc.png',
        )


def __plot_results_reg(actual: np.ndarray, expected: np.ndarray, path: str = None) -> None:
    """
    Generates and saves a residual plot for regression predictions.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values
    :param str path: Location where to save the generated plot

    :return: No returned data
    :rtype: None
    """

    diff = [y - x for x, y in zip(expected, actual)]
    x, y = zip(*sorted(zip(expected, diff)))
    trend_coeffs = np.polyfit(x, y, 1)
    trend_line = np.poly1d(trend_coeffs)(x)
    densities = gaussian_kde(np.vstack([y, x]))(np.vstack([y, x]))

    plot = plt.scatter(x, y, c=densities, s=25, cmap='coolwarm', label='_nolegend_')
    plt.axhline(y=0, color='b', linestyle='--')
    plt.plot(x, trend_line, color='g', linestyle='--')

    plt.title(None if path else config.datasets[config.dataset])
    plt.xlabel('Expected')
    plt.ylabel('Residuals')
    plt.ylim([-8, 8])
    plt.legend(['Origin', 'Trend Line'])
    plt.colorbar(plot)

    if path:
        plt.savefig(f'{path}/residual_plot.png')
    else:
        plt.savefig(
            f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/residual_plot.png',
        )


def plot_results(actual: np.ndarray, expected: np.ndarray, path: str = None) -> None:
    """
    Generates and saves a plots for predictions based on task.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values
    :param str path: Location where to save the generated plot

    :return: No returned data
    :rtype: None
    """

    if config.task == 'classification':
        __plot_results_cls(actual, expected, path)
    else:
        __plot_results_reg(actual, expected, path)
