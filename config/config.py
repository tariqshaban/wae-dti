import argparse
import datetime

import torch

experiment_start_time: int = int(datetime.datetime.now().timestamp() * 1000)
regression_datasets: dict[str, str] = {
    'davis': 'Davis',
    'kiba': 'Kiba',
    'dtc': 'DTC',
    'metz': 'Metz',
    'toxcast': 'ToxCast',
    'stitch': 'Stitch',
}
classification_datasets: dict[str, str] = {
    'biosnap': 'BioSNAP',
    'davis': 'Davis',
    'binding_db': 'BindingDB',
}
unknown_dataset: dict[str, str] = {
    'unknown': '',
}
datasets: dict[str, str] = regression_datasets | classification_datasets | unknown_dataset
drug_transformers: list[str] = [
    'AtomPairFingerprintTransformer',
    'AvalonFingerprintTransformer',
    'MACCSKeysFingerprintTransformer',
    'MHFingerprintTransformer',
    'MorganFingerprintTransformer',
    'RDKitFingerprintTransformer',
    'SECFingerprintTransformer',
    'TopologicalTorsionFingerprintTransformer',
    'LDP',
]

# Required for inference
models_path: str
input_file: str

drug_transformer: str | None
use_wandb: bool
task: str
dataset: str
seed: int | None
epochs: int
patience: int
learning_rate: float
eda: bool
batch_size: int
torch_device: str | None


def import_config(args: argparse.Namespace, inference: bool = False) -> None:
    """
    Import configuration settings from command line arguments into the global variables.

    :param argparse.Namespace args: The namespace to extract the arguments from
    :param bool inference: Specify whether the task is for training or inference

    :return: No returned data
    :rtype: None
    """

    global experiment_start_time
    global models_path
    global input_file
    global drug_transformer
    global use_wandb
    global task
    global dataset
    global seed
    global epochs
    global patience
    global learning_rate
    global eda
    global batch_size
    global torch_device

    if inference:
        models_path = args.models_path
        input_file = args.input_file
        dataset = 'unknown'
    else:
        if 'experiment_start_time' in vars(args).keys():
            experiment_start_time = args.experiment_start_time
        if 'drug_transformer' in vars(args).keys():
            drug_transformer = args.drug_transformer
        use_wandb = args.use_wandb
        dataset = args.dataset
        seed = args.seed
        epochs = args.epochs
        patience = args.patience
        learning_rate = args.learning_rate
        eda = args.eda

    task = args.task
    batch_size = args.batch_size
    torch_device = args.torch_device

    if not torch_device:
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
