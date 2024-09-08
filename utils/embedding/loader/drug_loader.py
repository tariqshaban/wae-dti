import numpy as np
import pandas as pd

from config import config


def load_drug_embedding() -> dict[np.ndarray, np.ndarray]:
    """
    Load drug embeddings from a parquet file.

    :return: A dictionary mapping drug identifiers to their corresponding embeddings
    :rtype: dict[np.ndarray, np.ndarray]
    """

    df = pd.read_parquet(f'data/embeddings/drug_embedding/{config.drug_transformer}.parquet')

    return dict(zip(df['drug'].to_numpy(), df['embedding'].to_numpy()))
