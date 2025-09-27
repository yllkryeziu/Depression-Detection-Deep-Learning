#  Depression Detection from Audio

This repository contains experiments for detecting depression from speech using deep learning models. The project leverages the `autrainer` toolkit, a modular framework built on PyTorch and Hydra for reproducible computer audition research.

The primary goal is to classify patients as depressive or non-depressive based on features extracted from their speech. Two main architectural approaches are explored:

1.  **CNN10 Model**: A Convolutional Neural Network trained directly on log-Mel spectrograms for binary classification.
2.  **CNN-LSTM Model**: Uses a pre-trained CNN10 as a feature extractor, followed by an LSTM with an attention mechanism to model temporal sequences.

## Table of Contents

- [Project Structure](#project-structure)
- [1. Setup and Installation](#1-setup-and-installation)
  - [1.1. Clone the Repository](#11-clone-the-repository)
  - [1.2. Create a Virtual Environment](#12-create-a-virtual-environment)
  - [1.3. Install Dependencies](#13-install-dependencies)
- [2. Environment Configuration](#2-environment-configuration)
  - [2.1. Create Environment File](#21-create-environment-file)
- [3. Data Preparation](#3-data-preparation)
  - [3.1. Fetch Data](#31-fetch-data)
  - [3.2. Preprocess Data](#32-preprocess-data)
- [4. Running Experiments](#4-running-experiments)
  - [Experiment 1: CNN10 Binary Classification](#experiment-1-cnn10-binary-classification)
  - [Experiment 2: CNN-LSTM Training](#experiment-2-cnn-lstm-training)
    - [Extract CNN Features](#extract-cnn-features)
    - [Train the LSTM Model](#train-the-lstm-model)
- [5. Post-processing and Analyzing Results](#5-post-processing-and-analyzing-results)
- [6. Inference](#6-inference)
- [7. Manual Evaluation](#7-manual-evaluation)

## Project Structure

The repository is organized to support reproducible experiments with `autrainer` and Hydra.

```
/
├── conf/              # Hydra/autrainer configuration files for datasets, models, etc.
├── data/              # Processed data (e.g., features, spectrograms).
├── data_raw/          # Raw data downloaded by fetch commands.
├── results/           # Experiment outputs, logs, and trained models.
├── requirements.txt   # Project dependencies.
├── postprocess.sh     # Helper script to analyze results.
└── README.md          # This file.
```

## 1. Setup and Installation

Follow these steps to set up the environment and install the necessary dependencies.

### 1.1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 1.2. Create a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

### 1.3. Install Dependencies

Install all required packages, including `autrainer` and its dependencies.

```bash
pip install -r requirements.txt
```

This project may require `openSMILE` for certain feature extraction configurations. If needed, install it separately and ensure it's available in your system's PATH. You can install the optional `opensmile` dependency for `autrainer` with:

```bash
pip install autrainer[opensmile]
```

## 2. Environment Configuration

Before running the experiments, you need to configure the dataset URL in an environment file.

### 2.1. Create Environment File

Copy the example environment file and fill in the Extended DAIC-WOZ dataset link:

```bash
cp .env.example .env
```

Then edit the `.env` file and replace `<LINK TO DATASET>` with the actual URL to the Extended DAIC-WOZ dataset. You can reference `.env.example` to see the expected format:

```bash
BASE_URL=<LINK TO DATASET>
```

## 3. Data Preparation

The experiments rely on the Extended DAIC-WOZ dataset. The following commands will download and preprocess the data into the required format.

### 3.1. Fetch Data

This command downloads the necessary datasets and pre-trained model weights specified in the configuration files.

```bash
autrainer fetch
```

### 3.2. Preprocess Data

This command processes the raw audio files into the features required for training (e.g., log-Mel spectrograms). The configurations in the `conf/` directory are set up to handle this automatically.

```bash
autrainer preprocess
```

## 4. Running Experiments

All experiments are managed by `autrainer` and can be launched from the command line. The results, including logs, model checkpoints, and plots, will be saved in the `results/` directory.

### Experiment 1: CNN10 Binary Classification

This experiment trains a CNN10 model directly on log-Mel spectrograms for end-to-end depression classification.

Run the baseline training with a fixed dataset split:

```bash
autrainer train -cn config-fixed
```

### Experiment 2: CNN-LSTM Training

This workflow involves two stages: extracting features with the CNN, and then training the LSTM on those features.

#### Extract CNN Features

First, run the feature extraction script. This uses the pre-trained CNN10 model to generate feature sequences for each patient and saves them in `data/ExtendedDAIC-lstm/features/`.

```bash
python extract_cnn_features.py \
    --data_path data/ExtendedDAIC-16k \
    --output_path data/ExtendedDAIC-lstm \
    --model_path model.pt
```

#### Train the LSTM Model

Once the features are extracted, train the LSTM model. This script runs a standalone training process and logs results to Weights & Biases.

```bash
python lstm_standalone.py
```

## 5. Post-processing and Analyzing Results

`autrainer` provides tools to analyze the results of your experiments, especially for grid searches.

To summarize the results of an experiment and aggregate across different seeds:

```bash
# Replace <experiment_id> with the one from your config (e.g., cnn10-fixed)
autrainer postprocess results/<experiment_id> --aggregate seed
```

This will generate summary CSVs and plots in the `results/<experiment_id>/summary/` directory. You can also use the helper script:

```bash
./postprocess.sh <experiment_id>
```

## 6. Inference

To run inference on new audio files using a trained model, use the `autrainer inference` command.

You need to point to a trained model directory, an input directory with audio files, and an output directory.

```bash
# Example command for a trained model from the 'cnn10-fixed' experiment
# Note: The path to the specific run may vary.
autrainer inference \
    results/cnn10-fixed/training/ExtendedDAIC-16k-fixed_CNN10-binary_Adam_0.0001_32_epoch_50_None_None_42/ \
    /path/to/your/input_audio/ \
    /path/to/your/output_predictions/ \
    --preprocess-cfg log_mel_16k \
    --device cuda:0
```

## 7. Manual Evaluation

The repository includes a script to manually calculate detailed metrics from prediction files. After generating `depression_predictions.csv` and `snippet_predictions.csv` (e.g., using `predict_depression.py`), you can get a full evaluation report.

```bash
python calculate_metrics.py
```