# Depression Detection from Audio

This repository contains a set of experiments for detecting depression from speech using deep learning models. The project leverages the `autrainer` toolkit, which is a modular framework built on PyTorch and Hydra for reproducible computer audition research.

The primary goal is to classify patients as depressive or non-depressive based on features extracted from their speech. Two main architectural approaches are explored:
1.  A **CNN10 model** trained directly on log-Mel spectrograms for binary classification.
2.  A **CNN-LSTM model** that uses a pre-trained CNN10 as a feature extractor, followed by an LSTM with an attention mechanism to model temporal sequences.

## Project Structure

The repository is organized to support reproducible experiments with `autrainer` and Hydra.

```
/
├── conf/              # Hydra/autrainer configuration files for datasets, models, etc.
├── data/              # Processed data (e.g., features, spectrograms).
├── data_raw/          # Raw data downloaded by the fetch command.
├── results/           # Experiment outputs, logs, and trained models.
├── src/               # Optional directory for custom Python source code.
├── requirements.txt   # Project dependencies.
├── fetch.sh           # Helper script to download data.
├── preprocess.sh      # Helper script to preprocess data.
├── train.sh           # Helper script to run training.
├── postprocess.sh     # Helper script to analyze results.
└── README.md          # This file.
```

## 1. Setup and Installation

Follow these steps to set up the environment and install the necessary dependencies.

**1.1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-name>
```

**1.2. Create a Virtual Environment**
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate
```

**1.3. Install Dependencies**
Install all required packages, including `autrainer` and its dependencies.
```bash
pip install -r requirements.txt
```
This project may require `openSMILE` for certain feature extraction configurations. If needed, install it separately and ensure it's available in your system's PATH. You can install the optional `opensmile` dependency for `autrainer` with:
```bash
pip install autrainer[opensmile]
```

## 2. Data Preparation

The experiments rely on the Extended DAIC-WOZ dataset. The following commands will download and preprocess the data into the required format.

**2.1. Fetch Data**
This command downloads the necessary datasets and pre-trained model weights specified in the configuration files.
```bash
autrainer fetch
```
Alternatively, you can use the provided shell script: `./fetch.sh`

**2.2. Preprocess Data**
This command processes the raw audio files into the features required for training (e.g., log-Mel spectrograms). The configurations in the `conf/` directory are set up to handle this automatically.
```bash
autrainer preprocess
```
Alternatively, you can use the provided shell script: `./preprocess.sh`

## 3. Running Experiments

All experiments are managed by `autrainer` and can be launched from the command line. The results, including logs, model checkpoints, and plots, will be saved in the `results/` directory.

### Experiment 1: CNN10 Binary Classification

This experiment trains a CNN10 model directly on log-Mel spectrograms for end-to-end depression classification. Several configurations are provided.

To run the baseline training with a fixed dataset split:
```bash
autrainer train -cn config-fixed
```

To run with data augmentation:
```bash
autrainer train -cn config-augmented
```

To run with a class-balanced dataset:
```bash
autrainer train -cn config-balanced
```

### Experiment 2: CNN-LSTM Training

This workflow involves two stages: extracting features with the CNN, and then training the LSTM on those features.

**3.2.1. Extract CNN Features**
First, run the feature extraction script. This uses the pre-trained CNN10 model to generate feature sequences for each patient and saves them in `data/ExtendedDAIC-lstm/features/`.
```bash
python extract_cnn_features.py \
    --data_path data/ExtendedDAIC-16k \
    --output_path data/ExtendedDAIC-lstm \
    --model_path model.pt
```

**3.2.2. Train the LSTM Model**
Once the features are extracted, train the LSTM model. This script runs a standalone training process and logs results to Weights & Biases.
```bash
python lstm_standalone.py
```

## 4. Post-processing and Analyzing Results

`autrainer` provides powerful tools to analyze the results of your experiments, especially for grid searches.

To summarize the results of an experiment and aggregate across different seeds:
```bash
# Replace <experiment_id> with the one from your config (e.g., cnn10-fixed)
autrainer postprocess results/<experiment_id> --aggregate seed
```
This will generate summary CSVs and plots in the `results/<experiment_id>/summary/` directory. You can also use the helper script: `./postprocess.sh <experiment_id>`.

## 5. Inference

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

## 6. Manual Evaluation

The repository includes a script to manually calculate detailed metrics from prediction files. After generating `depression_predictions.csv` and `snippet_predictions.csv` (e.g., using `predict_depression.py`), you can get a full evaluation report.

```bash
python calculate_metrics.py
```

```