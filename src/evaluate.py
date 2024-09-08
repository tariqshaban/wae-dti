# Acknowledgment: Some formulas are based on the DeepGLSTM paper
#
# Referenced paper DOI: https://doi.org/10.1137/1.9781611977172.82
# Referenced formulas:  https://github.com/MLlab4CS/DeepGLSTM/blob/main/utils.py

import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, average_precision_score


def __calculate_r_squared_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate the R-squared error.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values

    :return: R-squared error
    :rtype: float
    """

    actual = np.array(actual)
    expected = np.array(expected)
    actual_mean = np.mean(actual)
    expected_mean = np.mean(expected)

    mult = sum((expected - expected_mean) * (actual - actual_mean))
    mult = mult * mult

    actual_sq = sum((actual - actual_mean) * (actual - actual_mean))
    expected_sq = sum((expected - expected_mean) * (expected - expected_mean))

    return mult / float(actual_sq * expected_sq)


def __calculate_k(actual: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate the scaling factor k.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values

    :return: Scaling factor k
    :rtype: float
    """

    return sum(actual * expected) / float(sum(expected * expected))


def __calculate_squared_error_zero(actual: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate the squared error when the intercept is zero.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values

    :return: Squared error with zero intercept
    :rtype: float
    """

    k = __calculate_k(actual, expected)

    actual_mean = np.mean(actual)
    upp = sum((actual - (k * expected)) * (actual - (k * expected)))
    down = sum((actual - actual_mean) * (actual - actual_mean))

    return 1 - (upp / float(down))


def __calculate_rm2(actual: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate the rm2 metric.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values

    :return: rm2 metric
    :rtype: float
    """
    r2 = __calculate_r_squared_error(actual, expected)
    r02 = __calculate_squared_error_zero(actual, expected)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def __calculate_mse(actual: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate the mean squared error (MSE).

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values

    :return: Mean squared error
    :rtype: float
    """

    return ((actual - expected) ** 2).mean(axis=0)


def evaluate_cls(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Evaluate the predictions using sensitivity, specificity, AUC, AUPRC, and threshold metrics.
    Assumes the task to be classification.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values

    :return: Tuple containing sensitivity, specificity, AUC, AUPRC, and threshold
    :rtype: tuple[float, float, float]
    """

    fpr, tpr, thresholds = roc_curve(actual, expected)

    distances = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    optimal_index = np.argmin(distances)
    optimal_threshold = thresholds[optimal_index]

    expected_round = (expected >= optimal_threshold).astype(float)

    tn, fp, fn, tp = confusion_matrix(actual, expected_round).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    auc = roc_auc_score(actual, expected)
    auprc = average_precision_score(actual, expected)

    return sensitivity, specificity, auc, auprc, optimal_threshold


def evaluate_reg(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    """
    Evaluate the predictions using MSE, CI, and rm2 metrics.
    Assumes the task to be regression.

    :param np.ndarray actual: Array of actual values
    :param np.ndarray expected: Array of expected values

    :return: Tuple containing MSE, CI, and rm2
    :rtype: tuple[float, float, float]
    """

    c_index = concordance_index(actual, expected)
    rm2 = __calculate_rm2(actual, expected)
    mse = __calculate_mse(actual, expected)

    return mse, c_index, rm2
