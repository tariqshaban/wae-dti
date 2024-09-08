import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import config
from src.dti_model import DTIModel

os.environ['WANDB_SILENT'] = 'true'


def predict(model: DTIModel, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform predictions using the specified DTI model on the provided data loader.

    :param DTIModel model: The DTI model
    :param DataLoader loader: Validation or testing dataloader

    :return: A tuple containing numpy arrays of predictions and labels
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            drugs_batch = data[0].to(config.torch_device)
            targets_batch = data[1].to(config.torch_device)

            if config.task == 'classification':
                predictions_batch = torch.sigmoid(model(drugs_batch, targets_batch))
            else:
                predictions_batch = model(drugs_batch, targets_batch)
            labels_batch = torch.Tensor(data[2]).to(config.torch_device)

            predictions = torch.cat((predictions, predictions_batch.cpu()), 0)
            labels = torch.cat((labels, labels_batch.cpu()), 0)

    return predictions.numpy().flatten(), labels.numpy().flatten()
