import argparse
import os

import numpy as np
import pandas as pd
import torch
from termcolor import colored
from torch.utils.data import DataLoader

from config import config
from src.dti_dataset import DTIDataset, collate
from src.dti_model import DTIModel
from src.ensemble import apply_ensemble_external
from src.predict import predict
from utils.embedding.extractor.drug_extractor import extract_drugs
from utils.embedding.extractor.target_extractor import extract_targets

__model: DTIModel
__drugs: np.ndarray
__targets: np.ndarray
__labels: np.ndarray


def __setup_directories():
    os.makedirs(f'data/embeddings/drug_embedding', exist_ok=True)
    os.makedirs(f'data/embeddings/target_embedding', exist_ok=True)
    os.makedirs(f'data/embeddings/target_embedding_esm2', exist_ok=True)
    os.makedirs(f'{config.models_path}/_Ensemble', exist_ok=True)


def __read_input_file():
    global __drugs
    global __targets
    global __labels

    df = pd.read_csv(config.input_file)

    __drugs, __targets, __labels = (
        df['drug'].to_numpy(),
        df['target'].to_numpy(),
        df['label'].to_numpy() if 'label' in df else None,
    )


def __add_missing_embeddings():
    print(colored(f'Adding missing drug embeddings...', 'green'))
    print(colored('===========================================', 'green'))
    extract_drugs(__drugs)
    print('\n')
    print(colored(f'Adding missing target embeddings...', 'green'))
    print(colored('===========================================', 'green'))
    extract_targets(__targets)
    print('\n')


def __mount_model():
    global __model

    __model = DTIModel()
    __model.load_state_dict(
        torch.load(f'{config.models_path}/{config.drug_transformer}/model.pt', weights_only=False),
    )
    __model.to(config.torch_device)


def __get_predictions():
    test_data = DTIDataset(
        drugs=__drugs,
        targets=__targets,
        y=np.zeros(len(__drugs)) if __labels is None else __labels,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    predictions, _ = predict(__model, test_loader)

    return predictions


def __get_ensemble_predictions(predictions):
    print(colored('Ensemble', 'green'))
    print(colored('===========================================', 'green'))
    apply_ensemble_external(
        config.models_path,
        __drugs,
        __targets,
        predictions,
        __labels,
    )
    print(f'View the "{config.models_path}/_Ensemble" folder')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-path', type=str, required=True,
                        help='Folder path which contains the models trained on each drug fingerprint')
    parser.add_argument('--input-file', type=str, required=True,
                        help='CSV file path containing "drug", "target", and "label" columns (label is optional)')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True,
                        help='Training task (classification or regression)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Number of examples in each batch')
    parser.add_argument('--torch-device', type=str, help='Device used for training (e.g. cuda:0, cpu)')
    args = parser.parse_args()
    config.import_config(args, True)

    __setup_directories()
    __read_input_file()
    __add_missing_embeddings()
    predictions_dict = {}
    for drug_transformer in config.drug_transformers:
        config.drug_transformer = drug_transformer
        __mount_model()
        predictions_dict[config.drug_transformer] = __get_predictions()
    __get_ensemble_predictions(predictions_dict)
