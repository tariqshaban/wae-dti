import argparse
import os
import sys

import pandas as pd
from termcolor import colored

from config import config
from src.ensemble import apply_ensemble
from utils.eda import perform_eda
from utils.embedding.extractor.drug_extractor import extract_drugs
from utils.embedding.extractor.target_extractor import extract_targets
from utils.terminal_command_runner import run_terminal_command


def __eda():
    if config.eda and config.task == 'regression':
        perform_eda()


def __setup_directories():
    os.makedirs(f'data/embeddings/drug_embedding', exist_ok=True)
    os.makedirs(f'data/embeddings/target_embedding', exist_ok=True)
    os.makedirs(f'data/embeddings/target_embedding_esm2', exist_ok=True)
    for drug_transformer in config.drug_transformers:
        os.makedirs(f'results/{config.experiment_start_time}/{config.dataset}/{drug_transformer}', exist_ok=True)
    os.makedirs(f'results/{config.experiment_start_time}/{config.dataset}/_Ensemble', exist_ok=True)


def __add_missing_embeddings():
    train_data = pd.read_parquet(f'data/raw/{config.task}/{config.dataset}/train')
    if os.path.isdir(f'data/raw/{config.task}/{config.dataset}/valid'):
        valid_data = pd.read_parquet(f'data/raw/{config.task}/{config.dataset}/valid')
    else:
        valid_data = None
    test_data = pd.read_parquet(f'data/raw/{config.task}/{config.dataset}/test')
    complete_data = pd.concat([train_data, valid_data, test_data], axis=0)
    drugs = complete_data['compound_iso_smiles'].tolist()
    targets = complete_data['target_sequence'].tolist()

    print(colored(f'Adding missing drug embeddings...', 'green'))
    print(colored('===========================================', 'green'))
    extract_drugs(drugs)
    print('\n')
    print(colored(f'Adding missing target embeddings...', 'green'))
    print(colored('===========================================', 'green'))
    extract_targets(targets)
    print('\n')


def __train_models():
    print(colored(f'Training models...', 'green'))
    print(colored('===========================================', 'green'))
    for drug_transformer in config.drug_transformers:
        print()
        config.drug_transformer = drug_transformer
        print(f'{config.drug_transformer}')
        print('----------------------------------------')
        # Training has to be on a different runtime to ensure reproducibility
        args_string = ' '.join(f'--{key.replace("_", "-")} {value}' for key, value in vars(args).items() if value)
        args_string = (
            f'--experiment-start-time {config.experiment_start_time} '
            f'--drug-transformer {config.drug_transformer} '
            f'{args_string}'
        )
        command = f'{sys.executable} -u ./src/train.py {args_string}'
        run_terminal_command(command)

    print('\n')


def __get_ensemble_predictions():
    print(colored('Ensemble', 'green'))
    print(colored('===========================================', 'green'))
    apply_ensemble(f'results/{config.experiment_start_time}/{config.dataset}')
    print('View the "results" folder')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-wandb', type=bool, default=False, help='Log training and validation metrics into W&B')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True,
                        help='Training task (classification or regression)')
    parser.add_argument('--dataset', type=str, default='davis', help='Dataset used for training')
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

    __eda()
    __setup_directories()
    __add_missing_embeddings()
    __train_models()
    __get_ensemble_predictions()
