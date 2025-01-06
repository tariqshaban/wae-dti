import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import config
from src.dti_dataset import DTIDataset, collate
from src.dti_model import DTIModel
from src.evaluate import evaluate_reg, evaluate_cls
from src.predict import predict
from utils.results_plotter import plot_results


def test(drugs: np.ndarray, targets: np.ndarray, labels: np.ndarray) -> None:
    """
    Test the DTI model on provided drug-target pairs and evaluate performance.

    :param np.ndarray drugs: Array of drug identifiers
    :param np.ndarray targets: Array of target identifiers
    :param np.ndarray labels: Array of true labels corresponding to drug-target pairs

    :return: No returned data
    :rtype: None
    """

    def __seed_worker(_):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if config.seed:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.use_deterministic_algorithms(True)

    torch_generator = torch.Generator()
    if config.seed:
        torch_generator.manual_seed(config.seed)

    test_data = DTIDataset(
        drugs=drugs,
        targets=targets,
        y=labels,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
        worker_init_fn=__seed_worker,
        generator=torch_generator,
    )

    model = DTIModel()
    model.load_state_dict(
        torch.load(
            f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/model.pt',
            weights_only=False,
        ),
    )
    model.to(config.torch_device)

    expected, actual = predict(model, test_loader)

    if config.task == 'classification':
        test_sensitivity, test_specificity, test_auc, test_auprc, test_threshold = evaluate_cls(actual, expected)

        print(
            f'test_sensitivity={test_sensitivity:.3f},'
            f'test_specificity={test_specificity:.3f},'
            f'test_auc={test_auc:.3f},'
            f'test_auprc={test_auprc:.3f}',
            f'test_threshold={test_threshold:.3f}',
        )

        df_metrics = pd.read_csv(
            f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/metrics.csv',
        )
        df_metrics.insert(len(df_metrics.columns), 'test_sensitivity', test_sensitivity)
        df_metrics.insert(len(df_metrics.columns), 'test_specificity', test_specificity)
        df_metrics.insert(len(df_metrics.columns), 'test_auc', test_auc)
        df_metrics.insert(len(df_metrics.columns), 'test_auprc', test_auprc)
        df_metrics.insert(len(df_metrics.columns), 'test_threshold', test_threshold)
        df_metrics.to_csv(
            f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/metrics.csv',
            index=False,
        )
    else:
        test_mse, test_ci, test_rm2 = evaluate_reg(actual, expected)

        print(f'test_mse={test_mse:.3f}, test_ci={test_ci:.3f}, test_rm2={test_rm2:.3f}')

        df_metrics = pd.read_csv(
            f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/metrics.csv',
        )
        df_metrics.insert(len(df_metrics.columns), 'test_mse', test_mse)
        df_metrics.insert(len(df_metrics.columns), 'test_ci', test_ci)
        df_metrics.insert(len(df_metrics.columns), 'test_rm2', test_rm2)
        df_metrics.to_csv(
            f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/metrics.csv',
            index=False,
        )

    expected = expected.tolist()
    actual = actual.tolist()
    diff = [y - x for x, y in zip(expected, actual)]
    diff_abs = [abs(x) for x in diff]
    df_prediction = pd.DataFrame({
        'drug': drugs,
        'target': targets,
        'expected': expected,
        'actual': actual,
        'diff': diff,
        'diff_abs': diff_abs,
    })
    df_prediction.to_csv(
        f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/predictions.csv',
        index=False,
    )

    plot_results(actual, expected)
