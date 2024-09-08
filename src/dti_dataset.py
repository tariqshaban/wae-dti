import numpy as np
import torch
from torch.utils.data import Dataset

from utils.embedding.loader.drug_loader import load_drug_embedding
from utils.embedding.loader.target_loader import load_target_embedding


class DTIDataset(Dataset):
    """
    Dataset class for Drug-Target Interaction (DTI) data.

    :param np.ndarray drugs: Array of drug identifiers
    :param np.ndarray targets: Array of target identifiers
    :param np.ndarray y: Array of labels corresponding to drug-target pairs
    """

    def __init__(self, drugs: np.ndarray, targets: np.ndarray, y: np.ndarray):
        super().__init__()
        self.drugs = drugs
        self.targets = targets
        self.y = y
        self.drugs_embedding = None
        self.targets_embedding = None
        self.process()

    def process(self) -> None:
        """
        Process the dataset to load embeddings for drugs and targets.

        :return: No returned data
        :rtype: None

        :raises Exception: If drugs, targets, and y lengths do not match
        """

        if len(self.drugs) != len(self.targets) or len(self.drugs) != len(self.y):
            raise Exception('"drugs", "targets", and "y" must have the same length')

        drugs_embedding = []
        targets_embedding = []

        drug_dict = load_drug_embedding()
        target_dict = load_target_embedding()

        for i in range(len(self.drugs)):
            # The NumPy array needs to be copied so that it becomes writable; doing so will resolve this warning:
            # "UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors."
            drugs_embedding.append(torch.Tensor(drug_dict[self.drugs[i]].copy()))
            targets_embedding.append(torch.Tensor(target_dict[self.targets[i]].copy()))

        self.drugs_embedding = drugs_embedding
        self.targets_embedding = targets_embedding

    def __len__(self):
        """
        Return the size of the dataset.

        :return: Number of samples in the dataset
        :rtype: int
        """

        return len(self.drugs_embedding)

    def __getitem__(self, idx: int):
        """
        Retrieve a sample from the dataset.

        :param int idx: Index of the sample to retrieve

        :return: A tuple containing the drug embedding, target embedding, and label
        :rtype: tuple[torch.Tensor, torch.Tensor, float]
        """

        return self.drugs_embedding[idx], self.targets_embedding[idx], self.y[idx]


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, np.ndarray]]):
    """
    Custom collate function for DataLoader.

    :param batch: List of samples from the dataset

    :return: A tuple containing stacked drug embeddings, target embeddings, and labels
    :rtype: tuple[torch.Tensor, torch.Tensor, np.ndarray]
    """

    drugs = [item[0] for item in batch]
    drugs = torch.stack(drugs)
    targets = [item[1] for item in batch]
    targets = torch.stack(targets)
    labels = [item[2] for item in batch]

    return drugs, targets, labels
