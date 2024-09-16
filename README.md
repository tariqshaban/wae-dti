WAE-DTI: Ensemble-based architecture for drug-target interaction prediction using descriptors and embeddings
==============================


Getting Started
------------
Clone the project from GitHub and install the necessary dependencies.

```shell
git clone https://github.com/tariqshaban/wae-dti
cd wae-dti
pip install -r requirements.txt
```

Pytorch needs to be installed (preferably utilizing GPU acceleration "CUDA-enabled").

The program will attempt to clone the ESM repository if it is not present automatically. Either install Git on your
machine or manually download the repository and place it in ../esm.

> [!TIP]
> You can also clone the ESM repository beforehand by running the following command:
>
> ```shell
> git clone https://github.com/facebookresearch/esm ../esm
> ```


Project Structure
------------

    ├── config
    │   └── config.py                     <- Store terminal arguments from entry files.
    ├── data
    │   ├── embeddings
    │   │   ├── drug_embedding            <- Pre-trained drug embeddings (Parquet format).
    │   │   └── target_embedding          <- Pre-trained target embeddings (Parquet format).
    │   └── raw
    │       ├── classification            <- Raw classification datasets for training and evaluation (Parquet format).
    │       └── regression                <- Raw regression datasets for training and evaluation (Parquet format).
    ├── results                           <- Store trained model, predictions, and metrics.
    ├── saved_models
    │   ├── classification                <- Trained models and their performance on the classification task.
    │   └── regression                    <- Trained models and their performancee on the regression task.
    ├── src
    │   ├── dti_dataset.py                <- Data preprocessing and mounting prior to training.
    │   ├── dti_model.py                  <- Neural network definition for the WAE-DTI architecture.
    │   ├── ensemble.py                   <- Implementation of the weighted average ensemble method.
    │   ├── evaluate.py                   <- Mathematical methods to evaluate model performance.
    │   ├── predict.py                    <- Provide predictions of a model given a dataloader.
    │   ├── test.py                       <- Generate and evaluate predictions of the model given external examples.
    │   └── train.py                      <- WAE-DTI model trainer.
    ├── utils
    │   ├── embedding
    │   │   ├── extractor
    │   │   │   ├── drug_extractor.py     <- Extract and save embeddings from a list of drugs (Parquet format).
    │   │   │   └── target_extractor.py   <- Extract and save embeddings from a list of targets (Parquet format).
    │   │   └── loader
    │   │       ├── drug_loader.py        <- Load drug embeddings from a Parquet file.
    │   │       └── target_loader.py      <- Load target embeddings from a Parquet file.
    │   ├── eda.py                        <- Perform exploratory data analysis (EDA) on all datasets.
    │   └── terminal_command_runner.py    <- Run external terminal commands in real time.
    ├── entry_inference.py                <- Entry point for inference.
    ├── entry_train.py                    <- Entry point for training.
    ├── README.md                         <- README file and documentation.
    └── requirements.txt                  <- List of dependencies required to run the project.

Usage
------------

### Training

```shell
entry_train.py [-h] [--use-wandb USE_WANDB] --task {classification,regression} [--dataset DATASET] [--seed SEED] [--epochs EPOCHS] [--patience PATIENCE]
               [--eda EDA] [--learning-rate LEARNING_RATE] [--batch-size BATCH_SIZE] [--torch-device TORCH_DEVICE]
```

| Argument                        | Description                                                             | Default   | Notes                                                                                               |
|---------------------------------|-------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------------------------------------|
| `-h`, `--help`                  | Display a help message                                                  |           |                                                                                                     |
| `--use-wandb USE_WANDB`         | Log training and validation metrics into W&B                            | `False`   | If set to `True`, the code will initialize a project named by the selected dataset                  |
| `--task`                        | Supervised learning algorithm type                                      |           | Must be set to either `classification` or `regression`                                              |
| `--dataset DATASET`             | Dataset used for training                                               | `'davis'` | The dataset must be in the `data/raw` path with `train` and `test` folders containing Parquet files |
| `--seed SEED`                   | Ensures reproducibility (on the same device)                            |           |                                                                                                     |
| `--epochs EPOCHS`               | Number of training epochs                                               | `3000`    |                                                                                                     |
| `--patience PATIENCE`           | Number of epochs to elapse without improvement for the training to stop | `200`     |                                                                                                     |
| `--eda EDA`                     | Conduct a quick EDA on startup                                          | `False`   | EDA is applied to all datasets regardless of the specified `--dataset`                              |
| `--learning-rate LEARNING_RATE` | Learning rate                                                           | `5e-4`    |                                                                                                     |
| `--batch-size BATCH_SIZE`       | Number of examples in each batch                                        | `1024`    |                                                                                                     |
| `--torch-device TORCH_DEVICE`   | Device used for training (e.g. cuda:0, cpu)                             |           | If not specified, GPU will be utilized (if any)                                                     |

### Inference

```bash
entry_inference.py [-h] --models-path MODELS_PATH --input-file INPUT_FILE 
                   [--batch-size BATCH_SIZE] [--torch-device TORCH_DEVICE]
```

| Argument                      | Description                                                                        | Default | Notes                                                                                                                  |
|-------------------------------|------------------------------------------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------|
| `-h`, `--help`                | Display a help message                                                             |         |                                                                                                                        |
| `--models-path MODELS_PATH`   | Folder path which contains the models trained on each drug fingerprint             |         | You can use the pretrained models of any dataset within `models/saved_models`                                          |
| `--input-file INPUT_FILE`     | CSV file path containing "drug", "target", and "label" columns (label is optional) |         | If "label" column is specified, you must have enough examples to satisfy the concordance index calculation requirement |
| `--task`                      | Supervised learning algorithm type                                                 |         | Must be set to either `classification` or `regression`                                                                 |
| `--batch-size BATCH_SIZE`     | Number of examples in each batch                                                   | `1024`  |                                                                                                                        |
| `--torch-device TORCH_DEVICE` | Device used for training (e.g. cuda:0, cpu)                                        |         | If not specified, GPU will be utilized (if any)                                                                        |

> [!NOTE]
> When running `entry_train.py` and `entry_inference.py`, missing embeddings from drugs and targets are automatically
> extracted and saved into `data/embeddings`
>
> ```shell
> git clone https://github.com/facebookresearch/esm ../esm
> ```

> [!IMPORTANT]
> During training, you may notice that `tqdm` reports monotonically increasing `total` value, this is caused by taking
> into account the dynamic fluctuation of the early stopping counter. So, rather than always having the total value
> equal to the number of epochs, the progress bar adjusts to display the smallest number of epochs needed to finish the
> training, which translates to the following formula:
>
> `total = min(epochs, elapsed_epochs + patience - elapsed_patience)`

Results
------------

The following tables are the result of training the model using nine drug descriptors and one target embedding (ESM-2)
on six datasets while repeating the experiment five times. The values represent the mean and standard deviation of each
metric.

<table>
   <thead>
      <tr>
         <th rowspan="2">Model</th>
         <th colspan="3">Davis</th>
         <th colspan="3">Kiba</th>
      </tr>
      <tr>
         <th>MSE</th>
         <th>CI</th>
         <th>R2</th>
         <th>MSE</th>
         <th>CI</th>
         <th>R2</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>Atom pair fingerprint</td>
         <td>0.229 ± 0.006</td>
         <td>0.892 ± 0.002</td>
         <td>0.706 ± 0.007</td>
         <td>0.155 ± 0.002</td>
         <td>0.885 ± 0.001</td>
         <td>0.751 ± 0.006</td>
      </tr>
      <tr>
         <td>Avalon fingerprint</td>
         <td>0.213 ± 0.005</td>
         <td>0.897 ± 0.004</td>
         <td>0.718 ± 0.011</td>
         <td>0.150 ± 0.002</td>
         <td>0.883 ± 0.001</td>
         <td>0.772 ± 0.006</td>
      </tr>
      <tr>
         <td>MACCS keys fingerprint</td>
         <td>0.217 ± 0.002</td>
         <td>0.895 ± 0.004</td>
         <td>0.706 ± 0.007</td>
         <td>0.171 ± 0.002</td>
         <td>0.868 ± 0.003</td>
         <td>0.732 ± 0.013</td>
      </tr>
      <tr>
         <td>MH fingerprint</td>
         <td>0.217 ± 0.007</td>
         <td>0.894 ± 0.002</td>
         <td>0.709 ± 0.014</td>
         <td>0.158 ± 0.002</td>
         <td>0.883 ± 0.001</td>
         <td>0.757 ± 0.014</td>
      </tr>
      <tr>
         <td>Morgan fingerprint</td>
         <td>0.220 ± 0.005</td>
         <td>0.895 ± 0.003</td>
         <td>0.700 ± 0.011</td>
         <td>0.157 ± 0.003</td>
         <td>0.883 ± 0.001</td>
         <td>0.753 ± 0.014</td>
      </tr>
      <tr>
         <td>RDKit fingerprint</td>
         <td>0.222 ± 0.008</td>
         <td>0.895 ± 0.002</td>
         <td>0.711 ± 0.011</td>
         <td>0.154 ± 0.001</td>
         <td>0.885 ± 0.002</td>
         <td>0.751 ± 0.010</td>
      </tr>
      <tr>
         <td>SEC fingerprint</td>
         <td>0.219 ± 0.004</td>
         <td>0.893 ± 0.004</td>
         <td>0.711 ± 0.004</td>
         <td>0.158 ± 0.002</td>
         <td>0.883 ± 0.002</td>
         <td>0.756 ± 0.009</td>
      </tr>
      <tr>
         <td>Topological torsion fingerprint</td>
         <td>0.218 ± 0.007</td>
         <td>0.896 ± 0.002</td>
         <td>0.704 ± 0.012</td>
         <td>0.158 ± 0.004</td>
         <td>0.883 ± 0.003</td>
         <td>0.747 ± 0.015</td>
      </tr>
      <tr>
         <td>LDP</td>
         <td>0.310 ± 0.003</td>
         <td>0.853 ± 0.003</td>
         <td>0.597 ± 0.014</td>
         <td>0.389 ± 0.003</td>
         <td>0.756 ± 0.001</td>
         <td>0.422 ± 0.007</td>
      </tr>
      <tr>
         <td><strong>Ensemble</strong></td>
         <td><strong>0.190 ± 0.001</strong></td>
         <td><strong>0.915 ± 0.001</strong></td>
         <td><strong>0.745 ± 0.003</strong></td>
         <td><strong>0.127 ± 0.001</strong></td>
         <td><strong>0.899 ± 0.000</strong></td>
         <td><strong>0.778 ± 0.003</strong></td>
      </tr>
   </tbody>
</table>

<table>
   <thead>
      <tr>
         <th rowspan="2">Model</th>
         <th colspan="3">DTC</th>
         <th colspan="3">Metz</th>
      </tr>
      <tr>
         <th>MSE</th>
         <th>CI</th>
         <th>R2</th>
         <th>MSE</th>
         <th>CI</th>
         <th>R2</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>Atom pair fingerprint</td>
         <td>0.188 ± 0.004</td>
         <td>0.883 ± 0.001</td>
         <td>0.787 ± 0.009</td>
         <td>0.357 ± 0.004</td>
         <td>0.787 ± 0.002</td>
         <td>0.570 ± 0.005</td>
      </tr>
      <tr>
         <td>Avalon fingerprint</td>
         <td>0.177 ± 0.003</td>
         <td>0.883 ± 0.003</td>
         <td>0.812 ± 0.013</td>
         <td>0.325 ± 0.009</td>
         <td>0.798 ± 0.005</td>
         <td>0.616 ± 0.005</td>
      </tr>
      <tr>
         <td>MACCS keys fingerprint</td>
         <td>0.201 ± 0.003</td>
         <td>0.867 ± 0.003</td>
         <td>0.800 ± 0.004</td>
         <td>0.337 ± 0.008</td>
         <td>0.793 ± 0.003</td>
         <td>0.610 ± 0.005</td>
      </tr>
      <tr>
         <td>MH fingerprint</td>
         <td>0.193 ± 0.006</td>
         <td>0.877 ± 0.004</td>
         <td>0.781 ± 0.010</td>
         <td>0.358 ± 0.008</td>
         <td>0.787 ± 0.002</td>
         <td>0.575 ± 0.016</td>
      </tr>
      <tr>
         <td>Morgan fingerprint</td>
         <td>0.191 ± 0.004</td>
         <td>0.880 ± 0.002</td>
         <td>0.789 ± 0.015</td>
         <td>0.365 ± 0.003</td>
         <td>0.785 ± 0.001</td>
         <td>0.561 ± 0.004</td>
      </tr>
      <tr>
         <td>RDKit fingerprint</td>
         <td>0.179 ± 0.005</td>
         <td>0.885 ± 0.002</td>
         <td>0.806 ± 0.017</td>
         <td>0.330 ± 0.003</td>
         <td>0.796 ± 0.001</td>
         <td>0.610 ± 0.008</td>
      </tr>
      <tr>
         <td>SEC fingerprint</td>
         <td>0.191 ± 0.001</td>
         <td>0.881 ± 0.001</td>
         <td>0.789 ± 0.007</td>
         <td>0.364 ± 0.005</td>
         <td>0.785 ± 0.002</td>
         <td>0.563 ± 0.009</td>
      </tr>
      <tr>
         <td>Topological torsion fingerprint</td>
         <td>0.189 ± 0.003</td>
         <td>0.880 ± 0.001</td>
         <td>0.786 ± 0.007</td>
         <td>0.363 ± 0.002</td>
         <td>0.787 ± 0.001</td>
         <td>0.563 ± 0.006</td>
      </tr>
      <tr>
         <td>LDP</td>
         <td>0.434 ± 0.005</td>
         <td>0.768 ± 0.001</td>
         <td>0.565 ± 0.007</td>
         <td>0.532 ± 0.009</td>
         <td>0.723 ± 0.003</td>
         <td>0.418 ± 0.011</td>
      </tr>
      <tr>
         <td><strong>Ensemble</strong></td>
         <td><strong>0.143 ± 0.001</strong></td>
         <td><strong>0.898 ± 0.001</strong></td>
         <td><strong>0.839 ± 0.008</strong></td>
         <td><strong>0.284 ± 0.004</strong></td>
         <td><strong>0.813 ± 0.001</strong></td>
         <td><strong>0.676 ± 0.006</strong></td>
      </tr>
   </tbody>
</table>

<table>
   <thead>
      <tr>
         <th rowspan="2">Model</th>
         <th colspan="3">ToxCast</th>
         <th colspan="3">STITCH</th>
      </tr>
      <tr>
         <th>MSE</th>
         <th>CI</th>
         <th>R2</th>
         <th>MSE</th>
         <th>CI</th>
         <th>R2</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>Atom pair fingerprint</td>
         <td>0.326 ± 0.002</td>
         <td>0.914 ± 0.002</td>
         <td>0.553 ± 0.008</td>
         <td>1.140 ± 0.003</td>
         <td>0.751 ± 0.004</td>
         <td>0.389 ± 0.004</td>
      </tr>
      <tr>
         <td>Avalon fingerprint</td>
         <td>0.321 ± 0.003</td>
         <td>0.916 ± 0.001</td>
         <td>0.559 ± 0.009</td>
         <td>1.078 ± 0.006</td>
         <td>0.740 ± 0.006</td>
         <td>0.417 ± 0.003</td>
      </tr>
      <tr>
         <td>MACCS keys fingerprint</td>
         <td>0.324 ± 0.000</td>
         <td>0.914 ± 0.001</td>
         <td>0.561 ± 0.006</td>
         <td>1.120 ± 0.006</td>
         <td>0.708 ± 0.004</td>
         <td>0.407 ± 0.002</td>
      </tr>
      <tr>
         <td>MH fingerprint</td>
         <td><i>very high MSE</i></td>
         <td><i>very low CI</i></td>
         <td><i>very low rm2</i></td>
         <td><i>very high MSE</i></td>
         <td><i>very low CI</i></td>
         <td><i>very low rm2</i></td>
      </tr>
      <tr>
         <td>Morgan fingerprint</td>
         <td>0.333 ± 0.002</td>
         <td>0.911 ± 0.002</td>
         <td>0.540 ± 0.012</td>
         <td>1.122 ± 0.006</td>
         <td>0.763 ± 0.004</td>
         <td>0.398 ± 0.005</td>
      </tr>
      <tr>
         <td>RDKit fingerprint</td>
         <td>0.323 ± 0.002</td>
         <td>0.915 ± 0.001</td>
         <td>0.553 ± 0.005</td>
         <td>1.143 ± 0.008</td>
         <td>0.743 ± 0.005</td>
         <td>0.389 ± 0.006</td>
      </tr>
      <tr>
         <td>SEC fingerprint</td>
         <td>0.333 ± 0.003</td>
         <td>0.911 ± 0.002</td>
         <td>0.540 ± 0.009</td>
         <td>1.167 ± 0.006</td>
         <td>0.759 ± 0.005</td>
         <td>0.374 ± 0.007</td>
      </tr>
      <tr>
         <td>Topological torsion fingerprint</td>
         <td>0.331 ± 0.002</td>
         <td>0.913 ± 0.001</td>
         <td>0.541 ± 0.006</td>
         <td>1.163 ± 0.003</td>
         <td>0.754 ± 0.004</td>
         <td>0.378 ± 0.004</td>
      </tr>
      <tr>
         <td>LDP</td>
         <td>0.369 ± 0.001</td>
         <td>0.898 ± 0.000</td>
         <td>0.506 ± 0.002</td>
         <td>1.542 ± 0.003</td>
         <td>0.627 ± 0.002</td>
         <td>0.201 ± 0.001</td>
      </tr>
      <tr>
         <td><strong>Ensemble</strong></td>
         <td><strong>0.308 ± 0.001</strong></td>
         <td><strong>0.922 ± 0.000</strong></td>
         <td><strong>0.581 ± 0.003</strong></td>
         <td><strong>0.934 ± 0.004</strong></td>
         <td><strong>0.772 ± 0.001</strong></td>
         <td><strong>0.488 ± 0.003</strong></td>
      </tr>
   </tbody>
</table>

### Classification Task Results

In addition to the mentioned results for the regression task, three datasets were used to evaluate the model for the
classification task.

<table>
   <thead>
      <tr>
         <th rowspan="2">Model</th>
         <th colspan="5">BioSNAP</th>
      </tr>
      <tr>
         <th>Sensitivity</th>
         <th>Specificity</th>
         <th>AUC</th>
         <th>AUPRC</th>
         <th>Threshold</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>Atom pair fingerprint</td>
         <td>0.824 ± 0.008</td>
         <td>0.830 ± 0.019</td>
         <td>0.899 ± 0.005</td>
         <td>0.905 ± 0.002</td>
         <td>0.560 ± 0.251</td>
      </tr>
      <tr>
         <td>Avalon fingerprint</td>
         <td>0.859 ± 0.008</td>
         <td>0.872 ± 0.005</td>
         <td>0.932 ± 0.002</td>
         <td>0.933 ± 0.001</td>
         <td>0.582 ± 0.121</td>
      </tr>
      <tr>
         <td>MACCS keys fingerprint</td>
         <td><strong>0.873 ± 0.006</strong></td>
         <td><strong>0.884 ± 0.006</strong></td>
         <td><strong>0.942 ± 0.002</strong></td>
         <td><strong>0.945 ± 0.002</strong></td>
         <td>0.557 ± 0.126</td>
      </tr>
      <tr>
         <td>Morgan fingerprint</td>
         <td>0.789 ± 0.008</td>
         <td>0.830 ± 0.004</td>
         <td>0.880 ± 0.003</td>
         <td>0.892 ± 0.004</td>
         <td>0.701 ± 0.082</td>
      </tr>
      <tr>
         <td>RDKit fingerprint</td>
         <td>0.841 ± 0.002</td>
         <td>0.857 ± 0.009</td>
         <td>0.915 ± 0.003</td>
         <td>0.917 ± 0.003</td>
         <td>0.670 ± 0.186</td>
      </tr>
      <tr>
         <td>SEC fingerprint</td>
         <td>0.795 ± 0.006</td>
         <td>0.826 ± 0.011</td>
         <td>0.879 ± 0.001</td>
         <td>0.888 ± 0.002</td>
         <td>0.682 ± 0.053</td>
      </tr>
      <tr>
         <td>Topological torsion fingerprint</td>
         <td>0.807 ± 0.014</td>
         <td>0.834 ± 0.010</td>
         <td>0.888 ± 0.004</td>
         <td>0.896 ± 0.003</td>
         <td>0.442 ± 0.098</td>
      </tr>
      <tr>
         <td>LDP</td>
         <td>0.767 ± 0.010</td>
         <td>0.805 ± 0.012</td>
         <td>0.858 ± 0.003</td>
         <td>0.862 ± 0.001</td>
         <td>0.537 ± 0.047</td>
      </tr>
      <tr>
         <td><strong>Ensemble</strong></td>
         <td>0.862 ± 0.003</td>
         <td>0.871 ± 0.006</td>
         <td>0.935 ± 0.001</td>
         <td>0.943 ± 0.001</td>
         <td>0.478 ± 0.018</td>
      </tr>
   </tbody>
</table>

<table>
   <thead>
      <tr>
         <th rowspan="2">Model</th>
         <th colspan="5">Davis</th>
      </tr>
      <tr>
         <th>Sensitivity</th>
         <th>Specificity</th>
         <th>AUC</th>
         <th>AUPRC</th>
         <th>Threshold</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>Atom pair fingerprint</td>
         <td>0.892 ± 0.009</td>
         <td>0.853 ± 0.021</td>
         <td>0.929 ± 0.005</td>
         <td>0.408 ± 0.014</td>
         <td>0.559 ± 0.168</td>
      </tr>
      <tr>
         <td>Avalon fingerprint</td>
         <td>0.882 ± 0.013</td>
         <td>0.847 ± 0.005</td>
         <td>0.925 ± 0.005</td>
         <td>0.431 ± 0.014</td>
         <td>0.457 ± 0.082</td>
      </tr>
      <tr>
         <td>MACCS keys fingerprint</td>
         <td>0.890 ± 0.006</td>
         <td>0.862 ± 0.009</td>
         <td>0.929 ± 0.003</td>
         <td>0.415 ± 0.021</td>
         <td>0.564 ± 0.119</td>
      </tr>
      <tr>
         <td>Morgan fingerprint</td>
         <td>0.881 ± 0.019</td>
         <td>0.855 ± 0.017</td>
         <td>0.926 ± 0.003</td>
         <td>0.390 ± 0.016</td>
         <td>0.590 ± 0.289</td>
      </tr>
      <tr>
         <td>RDKit fingerprint</td>
         <td>0.876 ± 0.020</td>
         <td>0.844 ± 0.011</td>
         <td>0.924 ± 0.004</td>
         <td>0.416 ± 0.002</td>
         <td>0.485 ± 0.093</td>
      </tr>
      <tr>
         <td>SEC fingerprint</td>
         <td>0.879 ± 0.018</td>
         <td>0.852 ± 0.009</td>
         <td>0.926 ± 0.004</td>
         <td>0.398 ± 0.022</td>
         <td>0.561 ± 0.125</td>
      </tr>
      <tr>
         <td>Topological torsion fingerprint</td>
         <td>0.874 ± 0.022</td>
         <td>0.851 ± 0.007</td>
         <td>0.919 ± 0.005</td>
         <td>0.383 ± 0.009</td>
         <td>0.618 ± 0.048</td>
      </tr>
      <tr>
         <td>LDP</td>
         <td>0.797 ± 0.015</td>
         <td>0.728 ± 0.014</td>
         <td>0.827 ± 0.003</td>
         <td>0.239 ± 0.003</td>
         <td>0.537 ± 0.059</td>
      </tr>
      <tr>
         <td><strong>Ensemble</strong></td>
         <td><strong>0.900 ± 0.013</strong></td>
         <td><strong>0.867 ± 0.010</strong></td>
         <td><strong>0.936 ± 0.002</strong></td>
         <td><strong>0.474 ± 0.011</strong></td>
         <td>0.528 ± 0.055</td>
      </tr>
   </tbody>
</table>

<table>
   <thead>
      <tr>
         <th rowspan="2">Model</th>
         <th colspan="5">BindingDB</th>
      </tr>
      <tr>
         <th>Sensitivity</th>
         <th>Specificity</th>
         <th>AUC</th>
         <th>AUPRC</th>
         <th>Threshold</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>Atom pair fingerprint</td>
         <td>0.869 ± 0.009</td>
         <td>0.830 ± 0.015</td>
         <td>0.909 ± 0.011</td>
         <td>0.636 ± 0.014</td>
         <td>0.481 ± 0.054</td>
      </tr>
      <tr>
         <td>Avalon fingerprint</td>
         <td>0.883 ± 0.007</td>
         <td>0.834 ± 0.012</td>
         <td>0.918 ± 0.003</td>
         <td>0.651 ± 0.005</td>
         <td>0.457 ± 0.045</td>
      </tr>
      <tr>
         <td>MACCS keys fingerprint</td>
         <td><strong>0.886 ± 0.009</strong></td>
         <td>0.841 ± 0.006</td>
         <td>0.923 ± 0.001</td>
         <td>0.671 ± 0.004</td>
         <td>0.372 ± 0.082</td>
      </tr>
      <tr>
         <td>Morgan fingerprint</td>
         <td>0.831 ± 0.005</td>
         <td>0.794 ± 0.003</td>
         <td>0.879 ± 0.002</td>
         <td>0.597 ± 0.001</td>
         <td>0.468 ± 0.052</td>
      </tr>
      <tr>
         <td>RDKit fingerprint</td>
         <td>0.885 ± 0.004</td>
         <td>0.832 ± 0.007</td>
         <td>0.914 ± 0.001</td>
         <td>0.634 ± 0.007</td>
         <td>0.490 ± 0.076</td>
      </tr>
      <tr>
         <td>SEC fingerprint</td>
         <td>0.844 ± 0.012</td>
         <td>0.795 ± 0.006</td>
         <td>0.881 ± 0.004</td>
         <td>0.598 ± 0.002</td>
         <td>0.500 ± 0.049</td>
      </tr>
      <tr>
         <td>Topological torsion fingerprint</td>
         <td>0.876 ± 0.015</td>
         <td>0.822 ± 0.014</td>
         <td>0.908 ± 0.013</td>
         <td>0.615 ± 0.016</td>
         <td>0.518 ± 0.056</td>
      </tr>
      <tr>
         <td>LDP</td>
         <td>0.818 ± 0.013</td>
         <td>0.788 ± 0.019</td>
         <td>0.873 ± 0.006</td>
         <td>0.511 ± 0.008</td>
         <td>0.489 ± 0.057</td>
      </tr>
      <tr>
         <td><strong>Ensemble</strong></td>
         <td>0.885 ± 0.006</td>
         <td><strong>0.853 ± 0.007</strong></td>
         <td><strong>0.931 ± 0.001</strong></td>
         <td><strong>0.707 ± 0.005</strong></td>
         <td>0.517 ± 0.022</td>
      </tr>
   </tbody>
</table>

--------