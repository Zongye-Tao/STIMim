# STIMim: Missing-Structure-Aware Imputation for Multivariate Time Series

STIMim is a **missing-value imputation** technique for multivariate time series data, focusing on the joint learning of missing structure, variable correlation, and time dependencies. It significantly improves imputation performance, with an average improvement of **25.5%/25.0%/26.0%** in MAE/RMSE/MRE for random missingness and **15.5%/6.5%/15.2%** for structured missingness compared to state-of-the-art methods.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Setup](#setup)
- [Dependencies](#dependencies)
- [Datasets](#datasets)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing Paper Results](#reproducing-paper-results)


## Overview
STIMim combines **Missing-Structure Learning (MSL)**, a **Probabilistic Mask Constructor (PMC)**, and an **iTransformer** backbone (with Mamba) to align training masks with real test-time missingness and to jointly capture variable interactions and long-range temporal patterns.

## Key Features
- **Missing-Structure Learning (MSL)**: aligns the training mask distribution with the test-time missingness.
- **Probabilistic Mask Constructor (PMC)**: builds smooth, differentiable masks that preserve block-contiguity and co-missing structure.
- **Variable–Time Joint Modeling**: iTransformer + Mamba for cross-variable dependencies and long-horizon temporal modeling.
- **Robustness**: works well under both random and structural missingness.

## Environment & Dependencies (Pinned)
Python: `3.10`

requirements.txt (put these lines exactly):
h5py==3.12.1
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
six==1.16.0
torch==2.3.1
tqdm==4.66.2
tsdb==0.6.2
packaging==24.1
transformers==4.44.2
mamba-ssm==2.2.2

## Setup & Installation
- Create and activate an environment (pick one):

Conda:
- `conda create -n stimim python=3.10 -y`
- `conda activate stimim`

Venv:
- `python3.10 -m venv .venv`
- `source .venv/bin/activate` (Windows: `.\venv\Scripts\activate`)

- Install dependencies:
- Save the pinned list above to `requirements.txt` at the repo root.
- `pip install -r requirements.txt`

## Datasets
The project supports common multivariate time-series datasets and preprocessed `.h5` files (e.g., `Air Quality (Beijing)`, `Electricity`, `ETT(m1)`).

## Dataset Preparation
1) Download the original datasets:
- `bash data_downloading.sh`

2) Process the datasets:
- `cd dataset_generating_scripts`
- Open `gene_UCI_BeijingAirQuality_dataset.py` and set the missingness mode:
\ \ \ \ Set `missing_mode` from choices=\"random\", \"structural\".
- Generate processed data: `bash dataset_generating.sh`

3) Rename the processed `.h5` file:
- **Enter** `data` and **rename the current** `.h5` **to** `datasets.h5` (to make the training mode unambiguous).

4) Learn the missing-structure distribution (Softmax VAE):
- `python SoftmaxVAE2.py` -> produces `vae_model.pth`

5) Run the main pipeline:
- `python main.py`

## Training
- Default entry point: `main.py`. Hyperparameters are defined in `arguments.py` (e.g., `model=STIMim`, device, learning rate, etc.).
- Logs and checkpoints are saved to paths defined in `arguments.py`.
- Example (if you also provide a standalone trainer): `python train.py --model STIMim --epochs 100 --batch_size 64 --lr 1e-4`

## Evaluation
- Open `arguments.py`, find the model stage parameter (e.g., `default='train', help="Model stage"`) and change `default='train'` to `default='test'`.
- Save the file, then run: `python main.py`.
- The script will evaluate on the test set and report MAE / RMSE / MRE in logs and console.
- To impute only, set the stage to `'impute'` (if supported).
