import math

import torch
import torch.nn as nn

from utils.embedding.loader.drug_loader import load_drug_embedding


class DTIModel(torch.nn.Module):
    """
    A PyTorch neural network model for Drug-Target Interaction (DTI) prediction.

    :param float dropout_p: The dropout probability in the dropout layers (default is 0.2)
    """

    def __init__(self, dropout_p: float = 0.2):
        super(DTIModel, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

        def closest_power_of_2(x: int):
            n = math.ceil(math.log2(x)) - 1
            return 2 ** n

        self.drugs_fcs = []

        input_size = len(list(load_drug_embedding().values())[0])
        closest_number = input_size

        while closest_number != 128:
            next_closest_number = closest_power_of_2(closest_number)
            self.drugs_fcs.append(nn.Linear(closest_number, next_closest_number))
            closest_number = next_closest_number
        self.drugs_fcs = nn.ModuleList(self.drugs_fcs)

        self.fc1 = nn.Linear(128 + 2560, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor):
        """
        Forward pass of the DTI model.

        :param torch.Tensor drug: The drug one-dimensional embedding tensor
        :param torch.Tensor protein: The protein one-dimensional embedding tensor

        :return: The output layer of the model
        :rtype: torch.Tensor
        """

        drug = drug
        for drug_fc in self.drugs_fcs:
            drug = drug_fc(drug)
            drug = self.relu(drug)
            drug = self.dropout(drug)

        xc = torch.cat((drug, protein), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
