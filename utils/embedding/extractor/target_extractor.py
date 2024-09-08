import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch

from utils.terminal_command_runner import run_terminal_command


def extract_targets(targets: np.ndarray) -> None:
    """
    Extract embeddings for targets and save them to a parquet file.

    :param np.ndarray targets: List of target identifiers to extract embeddings for

    :return: No returned data
    :rtype: None

    :raises Exception: If ESM2 failed to return valid embeddings
    """

    existing_target_dict = {}

    if os.path.isfile('data/embeddings/target_embedding/TargetTransformer.parquet'):
        df = pd.read_parquet('data/embeddings/target_embedding/TargetTransformer.parquet')
        existing_target_dict = dict(zip(df['target'], df['embedding']))

    targets = list(set(targets))
    targets = [x for x in targets if x not in existing_target_dict]

    if not targets:
        shutil.rmtree(f'data/embeddings/target_embedding_esm2')
        print('No missing target embeddings')
        return

    print(f'Found ({len(targets)}) missing target embeddings')

    incremental_target_dict = {idx: value for idx, value in enumerate(list(set(targets)))}

    result = ''
    for key, value in incremental_target_dict.items():
        result += f'>UniRef50_{key}\n{value}\n'
    with open(f'data/embeddings/targets.fasta', 'w') as f:
        f.write(result)

    if not os.path.isfile(f'../esm/scripts/extract.py'):
        command = 'git clone https://github.com/facebookresearch/esm.git ../esm'
        run_terminal_command(command)

    command = (f'{sys.executable} -u ../esm/scripts/extract.py '
               'esm2_t36_3B_UR50D '
               f'data/embeddings/targets.fasta '
               f'data/embeddings/target_embedding_esm2 '
               '--include mean per_tok'
               )

    run_terminal_command(command)

    esm_emb_path = f'data/embeddings/target_embedding_esm2'
    target_dict = {}

    try:
        for key in incremental_target_dict.keys():
            target_dict[incremental_target_dict[key]] = \
                torch.load(f'{esm_emb_path}/UniRef50_{key}.pt', weights_only=False)['mean_representations'][36]
    except FileNotFoundError:
        raise MemoryError(
            'The ESM2 extractor did not output any embeddings; '
            'if the extractor did not raise an explicit exception, '
            'ensure you have enough memory or use a less complex pretrained model.'
        )

    target_dict = {key: value.numpy() for key, value in target_dict.items()}

    target_dict = {**existing_target_dict, **target_dict}

    df = pd.DataFrame(list(target_dict.items()), columns=['target', 'embedding'])
    df.to_parquet('data/embeddings/target_embedding/TargetTransformer.parquet')

    os.remove(f'data/embeddings/targets.fasta')
    shutil.rmtree(f'data/embeddings/target_embedding_esm2')
