import numpy as np
import pandas as pd

from config import config
from src.evaluate import evaluate_reg, evaluate_cls
from utils.results_plotter import plot_results


def apply_ensemble(
        models_path: str,
) -> None:
    """
    Apply ensemble learning to generate predictions by averaging the predictions of multiple models.

    :param str models_path: Directory containing the model subdirectories with metrics and predictions CSV files

    :return: No returned data
    :rtype: None
    """

    punishment_factor: int = 6
    drugs: list[str] = []
    targets: list[float] = []
    actual: list[float] = []
    models_expected: dict[str, float] = {}
    models_weights: dict[str, float] = {}

    for drug_transformer in config.drug_transformers:
        df_metrics = pd.read_csv(f'{models_path}/{drug_transformer}/metrics.csv')
        df_predictions = pd.read_csv(f'{models_path}/{drug_transformer}/predictions.csv')
        drugs = df_predictions['drug'].tolist()
        targets = df_predictions['target'].tolist()
        expected = df_predictions['expected'].tolist()
        actual = df_predictions['actual'].tolist()
        models_expected[drug_transformer] = expected
        if config.task == 'classification':
            val_auprc = df_metrics['val_auprc']
            models_weights[drug_transformer] = val_auprc.iloc[0] ** punishment_factor
        else:
            val_mse = df_metrics['val_mse']
            models_weights[drug_transformer] = 1 / val_mse.iloc[0] ** punishment_factor

    print(f'Generating ensemble predictions from {len(config.drug_transformers)} models...')

    ensemble_expected = np.average(
        list(models_expected.values()),
        weights=list(models_weights.values()),
        axis=0,
    )

    if config.task == 'classification':
        test_sensitivity, test_specificity, test_auc, test_auprc, test_threshold = evaluate_cls(
            np.array(actual),
            ensemble_expected,
        )

        print(
            f'test_sensitivity={test_sensitivity:.3f},'
            f'test_specificity={test_specificity:.3f},'
            f'test_auc={test_auc:.3f},'
            f'test_auprc={test_auprc:.3f}',
            f'test_threshold={test_threshold:.3f}',
        )

        df_metrics = pd.DataFrame({
            'test_sensitivity': [test_sensitivity],
            'test_specificity': [test_specificity],
            'test_auc': [test_auc],
            'test_auprc': [test_auprc],
            'test_threshold': [test_threshold],
        })
    else:
        test_mse, test_ci, test_rm2 = evaluate_reg(np.array(actual), ensemble_expected)

        print(f'test_mse={test_mse:.3f}, test_ci={test_ci:.3f}, test_rm2={test_rm2:.3f}')

        df_metrics = pd.DataFrame({
            'test_mse': [test_mse],
            'test_ci': [test_ci],
            'test_rm2': [test_rm2],
        })

    df_metrics.to_csv(f'{models_path}/_Ensemble/metrics.csv', index=False)

    diff = [y - x for x, y in zip(ensemble_expected, actual)]
    diff_abs = [abs(x) for x in diff]
    df_prediction = pd.DataFrame({
        'drug': drugs,
        'target': targets,
        'expected': ensemble_expected,
        'actual': actual,
        'diff': diff,
        'diff_abs': diff_abs,
    })
    df_prediction.to_csv(f'{models_path}/_Ensemble/predictions.csv', index=False)

    plot_results(np.array(actual), ensemble_expected, f'{models_path}/_Ensemble')


def apply_ensemble_external(
        models_path: str,
        drugs: np.ndarray,
        targets: np.ndarray,
        expected: dict[str, float],
        labels: np.ndarray | None,
) -> None:
    """
    Apply ensemble learning to generate predictions using external data by averaging the predictions of multiple models.

    :param str models_path: Directory containing the model subdirectories with metrics and predictions CSV files
    :param np.ndarray drugs: Array of drug identifiers
    :param np.ndarray targets: Array of target identifiers
    :param dict[str, list[float]] expected: Dictionary of expected predictions from each embedding model
    :param np.ndarray | None labels: Optional array of actual labels for evaluation

    :return: No returned data
    :rtype: None
    """

    punishment_factor: int = 6
    models_weights: dict[str, float] = {}

    for drug_transformer in config.drug_transformers:
        df_metrics = pd.read_csv(f'{models_path}/{drug_transformer}/metrics.csv')
        if config.task == 'classification':
            val_auprc = df_metrics['val_auprc']
            models_weights[drug_transformer] = val_auprc.iloc[0] ** punishment_factor
        else:
            val_mse = df_metrics['val_mse']
            models_weights[drug_transformer] = 1 / val_mse.iloc[0] ** punishment_factor

    print(f'Generating ensemble predictions from {len(config.drug_transformers)} models...')

    ensemble_expected = np.average(
        list(expected.values()),
        weights=list(models_weights.values()),
        axis=0,
    )

    if labels is not None:
        if config.task == 'classification':
            test_sensitivity, test_specificity, test_auc, test_auprc, test_threshold = evaluate_cls(
                np.array(labels),
                ensemble_expected,
            )

            print(
                f'test_sensitivity={test_sensitivity:.3f},'
                f'test_specificity={test_specificity:.3f},'
                f'test_auc={test_auc:.3f},'
                f'test_auprc={test_auprc:.3f}',
                f'test_threshold={test_threshold:.3f}',
            )

            df_metrics = pd.DataFrame({
                'test_sensitivity': [test_sensitivity],
                'test_specificity': [test_specificity],
                'test_auc': [test_auc],
                'test_auprc': [test_auprc],
                'test_threshold': [test_threshold],
            })
        else:
            test_mse, test_ci, test_rm2 = evaluate_reg(np.array(labels), ensemble_expected)

            print(f'test_mse={test_mse:.3f}, test_ci={test_ci:.3f}, test_rm2={test_rm2:.3f}')

            df_metrics = pd.DataFrame({
                'test_mse': [test_mse],
                'test_ci': [test_ci],
                'test_rm2': [test_rm2],
            })

        df_metrics.to_csv(f'{models_path}/_Ensemble/metrics.csv', index=False)

    if labels is not None:
        diff = [y - x for x, y in zip(ensemble_expected, labels)]
        diff_abs = [abs(x) for x in diff]
        df_prediction = pd.DataFrame({
            'drug': drugs,
            'target': targets,
            'expected': ensemble_expected,
            'actual': labels,
            'diff': diff,
            'diff_abs': diff_abs,
        })
        df_prediction.to_csv(f'{models_path}/_Ensemble/predictions.csv', index=False)

        plot_results(labels, ensemble_expected, f'{models_path}/_Ensemble')
    else:
        df_prediction = pd.DataFrame({
            'drug': drugs,
            'target': targets,
            'expected': ensemble_expected,
        })
        df_prediction.to_csv(f'{models_path}/_Ensemble/predictions.csv', index=False)
