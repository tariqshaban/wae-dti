import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from src.dti_dataset import DTIDataset, collate
from src.dti_model import DTIModel
from src.evaluate import evaluate_reg, evaluate_cls
from src.predict import predict
from src.test import test

os.environ['WANDB_SILENT'] = 'true'

scaler = torch.amp.GradScaler()


def __train_epoch(model: DTIModel, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> float:
    """
    Train the model for a single epoch.

    :param DTIModel model: The DTI model to train
    :param DataLoader train_loader: DataLoader for the training data
    :param torch.optim optimizer: Optimizer for the model parameters
    :param epoch epoch: Current epoch number (for logging purposes)

    :return: Average loss for the epoch
    :rtype: float
    """

    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss() if config.task == 'classification' else torch.nn.MSELoss()
    average_loss = 0

    for batch_idx, data in enumerate(train_loader):
        drug_batch = torch.Tensor(data[0]).to(config.torch_device)
        target_batch = torch.Tensor(data[1]).to(config.torch_device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=config.torch_device):
            output = model(drug_batch, target_batch)
            labels = torch.Tensor(data[2]).to(config.torch_device)
            loss = loss_fn(output.flatten(), labels.flatten())
        average_loss += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    average_loss /= len(train_loader)
    wandb.log({'train_loss': average_loss}, step=epoch, commit=False)

    return average_loss


def __train_loop_cls(train_loader: DataLoader, valid_loader: DataLoader) -> None:
    """
    Train iterator for the classification task.

    :param DataLoader train_loader: DataLoader for the training data
    :param DataLoader valid_loader: DataLoader for the validation data

    :return: No returned data
    :rtype: None
    """

    model = DTIModel()
    model.to(config.torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)

    best_val_auprc = 0
    patience_counter = 0
    with tqdm(total=min(config.epochs, config.patience)) as pbar:
        for epoch in range(config.epochs):
            loss = __train_epoch(model, train_loader, optimizer, epoch + 1)
            expected, actual = predict(model, valid_loader)
            val_sensitivity, val_specificity, val_auc, val_auprc, val_threshold = evaluate_cls(actual, expected)
            wandb.log({'val_sensitivity': val_sensitivity}, step=epoch + 1, commit=False)
            wandb.log({'val_specificity': val_specificity}, step=epoch + 1, commit=False)
            wandb.log({'val_auc': val_auc}, step=epoch + 1, commit=False)
            wandb.log({'val_auprc': val_auprc}, step=epoch + 1, commit=False)
            wandb.log({'val_threshold': val_threshold}, step=epoch + 1)
            if best_val_auprc < val_auprc:
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/model.pt',
                )
                df_metrics = pd.DataFrame({
                    'val_sensitivity': [val_sensitivity],
                    'val_specificity': [val_specificity],
                    'val_auc': [val_auc],
                    'val_auprc': [val_auprc],
                    'val_threshold': [val_threshold],
                })
                df_metrics.to_csv(
                    f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/metrics.csv',
                    index=False,
                )
                best_val_auprc = val_auprc
            pbar.total = min(config.epochs, epoch + 1 + config.patience - patience_counter)
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss:.3f}',
                'val_sensitivity': f'{val_sensitivity:.3f}',
                'val_specificity': f'{val_specificity:.3f}',
                'val_auc': f'{val_auc:.3f}',
                'val_auprc': f'{val_auprc:.3f}',
                'val_threshold': f'{val_threshold:.3f}',
            })
            if patience_counter == config.patience:
                break
            patience_counter = patience_counter + 1


def __train_loop_reg(train_loader: DataLoader, valid_loader: DataLoader) -> None:
    """
    Train iterator for the regression task.

    :param DataLoader train_loader: DataLoader for the training data
    :param DataLoader valid_loader: DataLoader for the validation data

    :return: No returned data
    :rtype: None
    """

    model = DTIModel()
    model.to(config.torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)

    best_val_mse = float('inf')
    patience_counter = 0
    with tqdm(total=min(config.epochs, config.patience)) as pbar:
        for epoch in range(config.epochs):
            loss = __train_epoch(model, train_loader, optimizer, epoch + 1)
            expected, actual = predict(model, valid_loader)
            val_mse, val_ci, val_rm2 = evaluate_reg(actual, expected)
            wandb.log({'val_mse': val_mse}, step=epoch + 1, commit=False)
            wandb.log({'val_ci': val_ci}, step=epoch + 1, commit=False)
            wandb.log({'val_rm2': val_rm2}, step=epoch + 1)
            if best_val_mse > val_mse:
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/model.pt',
                )
                df_metrics = pd.DataFrame({
                    'val_mse': [val_mse],
                    'val_ci': [val_ci],
                    'val_rm2': [val_rm2],
                })
                df_metrics.to_csv(
                    f'results/{config.experiment_start_time}/{config.dataset}/{config.drug_transformer}/metrics.csv',
                    index=False,
                )
                best_val_mse = val_mse
            pbar.total = min(config.epochs, epoch + 1 + config.patience - patience_counter)
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss:.3f}',
                'val_mse': f'{val_mse:.3f}',
                'val_ci': f'{val_ci:.3f}',
                'val_rm2': f'{val_rm2:.3f}',
            })
            if patience_counter == config.patience:
                break
            patience_counter = patience_counter + 1


def __train_model() -> None:
    """
    Train the DTI model.

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

    wandb.init(
        project=config.dataset,
        name=config.drug_transformer,
        mode='online' if config.use_wandb else 'disabled',
        config={
            'seed': config.seed,
            'epochs': config.epochs,
            'patience': config.patience,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
        },
    )

    torch_generator = torch.Generator()
    if config.seed:
        torch_generator.manual_seed(config.seed)

    print(f'Loading dataset...')

    df_train = pd.read_parquet(f'data/raw/{config.task}/{config.dataset}/train')
    df_test = pd.read_parquet(f'data/raw/{config.task}/{config.dataset}/test')

    train_drugs, train_targets, train_y = (
        df_train['compound_iso_smiles'].to_numpy(),
        df_train['target_sequence'].to_numpy(),
        df_train['affinity'].to_numpy(),
    )

    test_drugs, test_targets, test_y = (
        df_test['compound_iso_smiles'].to_numpy(),
        df_test['target_sequence'].to_numpy(),
        df_test['affinity'].to_numpy(),
    )

    train_dataset = DTIDataset(
        drugs=train_drugs,
        targets=train_targets,
        y=train_y,
    )

    if os.path.isdir(f'data/raw/{config.task}/{config.dataset}/valid'):
        df_valid = pd.read_parquet(f'data/raw/{config.task}/{config.dataset}/valid')
        valid_drugs, valid_targets, valid_y = (
            df_valid['compound_iso_smiles'].to_numpy(),
            df_valid['target_sequence'].to_numpy(),
            df_valid['affinity'].to_numpy(),
        )
        valid_dataset = DTIDataset(
            drugs=valid_drugs,
            targets=valid_targets,
            y=valid_y,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
            worker_init_fn=__seed_worker,
            generator=torch_generator,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate,
            worker_init_fn=__seed_worker,
            generator=torch_generator,
        )
    else:
        train_dataset, valid_dataset = train_test_split(train_dataset, shuffle=True, test_size=0.2)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
            worker_init_fn=__seed_worker,
            generator=torch_generator,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate,
            worker_init_fn=__seed_worker,
            generator=torch_generator,
        )

    print(f'Training model (<={config.epochs} epochs)...')

    if config.task == 'classification':
        __train_loop_cls(train_loader, valid_loader)
    else:
        __train_loop_reg(train_loader, valid_loader)

    test(test_drugs, test_targets, test_y)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-start-time', type=str, help='Unix timestamp denoting the experiment start time')
    parser.add_argument('--drug-transformer', type=str, help='Drug transformer used in training')
    parser.add_argument('--use-wandb', type=bool, default=False, help='Log training and validation metrics into W&B')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True,
                        help='Training task (classification or regression)')
    parser.add_argument('--dataset', type=str, default='toxcast', help='Dataset used for training')
    parser.add_argument('--seed', type=int, help='Ensures reproducibility')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='Number of epochs to elapse without improvement for the training to stop')
    parser.add_argument('--eda', type=bool, default=False, help='Conduct a quick EDA on startup (regression only)')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1024, help='Number of examples in each batch')
    parser.add_argument('--torch-device', type=str, help='Device used for training (e.g. cuda:0, cpu)')
    args = parser.parse_args()
    config.import_config(args)

    __train_model()
