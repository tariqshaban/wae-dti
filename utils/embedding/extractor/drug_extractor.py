import os.path

import networkx as nx
import numpy as np
import pandas as pd
# noinspection PyProtectedMember
from karateclub import (
    Estimator,
    LDP,
)
from rdkit import Chem
from scikit_mol.fingerprints import (
    FpsTransformer,
    MorganFingerprintTransformer,
    MACCSKeysFingerprintTransformer,
    RDKitFingerprintTransformer,
    AtomPairFingerprintTransformer,
    TopologicalTorsionFingerprintTransformer,
    MHFingerprintTransformer,
    SECFingerprintTransformer,
    AvalonFingerprintTransformer,
)
from tqdm import tqdm


def __smiles_to_embedding(smiles_list: np.ndarray, transformer: FpsTransformer) -> np.ndarray:
    """
    Convert SMILES strings to embeddings using a fingerprint transformer.

    :param np.ndarray smiles_list: List of SMILES strings representing chemical compounds
    :param FpsTransformer transformer: Fingerprint transformer for generating embeddings

    :return: Normalized embeddings for the provided SMILES strings.
    :rtype: np.ndarray
    """

    smiles_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    parameters = transformer.get_params()
    transformer.set_params(**parameters)

    # noinspection PyTypeChecker
    embedding = transformer.transform(smiles_list)
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding))
    return embedding


def __smiles_to_embedding_karate_club(smiles_list: np.ndarray, transformer: Estimator) -> np.ndarray:
    """
    Convert SMILES strings to embeddings using a Karate Club estimator.

    :param np.ndarray smiles_list: List of SMILES strings representing chemical compounds
    :param Estimator transformer: Karate Club estimator for generating embeddings

    :return: Normalized embeddings for the provided SMILES strings
    :rtype: np.ndarray
    """

    smiles_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    graphs = []

    for smiles in smiles_list:
        g = nx.Graph()

        for atom in smiles.GetAtoms():
            g.add_node(
                atom.GetIdx(),
                atomic_num=atom.GetAtomicNum(),
                is_aromatic=atom.GetIsAromatic(),
                atom_symbol=atom.GetSymbol(),
            )

        for bond in smiles.GetBonds():
            g.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=bond.GetBondType(),
            )

        graphs.append(g)

    model = transformer
    # noinspection PyArgumentList
    model.fit(graphs)
    embedding = model.get_embedding()
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding))
    return embedding


def __drug_transformer(drugs: np.ndarray, transformer: FpsTransformer | Estimator) -> None:
    """
    Transform drugs into embeddings and save them to a parquet file.

    :param np.ndarray drugs: List of drug SMILES strings
    :param FpsTransformer | Estimator transformer: Fingerprint transformer or Karate Club estimator

    :return: No returned data
    :rtype: None
    """

    transformer_name = type(transformer).__name__
    existing_drug_dict = {}

    if os.path.isfile(f'data/embeddings/drug_embedding/{transformer_name}.parquet'):
        df = pd.read_parquet(f'data/embeddings/drug_embedding/{transformer_name}.parquet')
        existing_drug_dict = dict(zip(df['drug'], df['embedding']))

    drugs = np.unique(drugs)
    drugs = np.setdiff1d(drugs, np.array(list(existing_drug_dict.keys())))

    if drugs.size == 0:
        print(f'No missing drug embeddings for {transformer_name}')
        return

    print(f'Found ({len(drugs)}) missing drug embeddings for {transformer_name}')

    if isinstance(transformer, FpsTransformer):
        embedding = __smiles_to_embedding(drugs, transformer)
    else:
        embedding = __smiles_to_embedding_karate_club(drugs, transformer)

    drug_dict = dict(zip(drugs, embedding))

    drug_dict = {**existing_drug_dict, **drug_dict}

    df = pd.DataFrame(list(drug_dict.items()), columns=['drug', 'embedding'])
    df.to_parquet(f'data/embeddings/drug_embedding/{transformer_name}.parquet')


def extract_drugs(drugs: np.ndarray) -> None:
    """
    Extract embeddings for a list of drugs using various transformers.

    :param np.ndarray drugs: List of drug identifiers to extract embeddings for

    :return: No returned data
    :rtype: None
    """

    drug_transformers = [
        AtomPairFingerprintTransformer(),
        AvalonFingerprintTransformer(),
        MACCSKeysFingerprintTransformer(),
        MHFingerprintTransformer(),
        MorganFingerprintTransformer(),
        RDKitFingerprintTransformer(),
        SECFingerprintTransformer(),
        TopologicalTorsionFingerprintTransformer(),
        LDP(),
    ]

    for drug_transformer in tqdm(drug_transformers):
        __drug_transformer(drugs, drug_transformer)
