import numpy as np
import pandas as pd


def load_target_embedding() -> dict[np.ndarray, np.ndarray]:
    """
    Load target embeddings from a parquet file.

    :return: A dictionary mapping target identifiers to their corresponding embeddings
    :rtype: dict[np.ndarray, np.ndarray]
    """

    df = pd.read_parquet('data/embeddings/target_embedding/TargetTransformer.parquet')

    return dict(zip(df['target'].to_numpy(), df['embedding'].to_numpy()))
