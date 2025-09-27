import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration ---
# All hyperparameters are stored in a dictionary for easy logging to WANDB
config = {
    "project_name": "LSTM-v1",
    "data_dir": "data/ExtendedDAIC-lstm/features/",
    "train_labels_path": "data/ExtendedDAIC-lstm/train_patients.csv",
    "val_labels_path": "data/ExtendedDAIC-lstm/dev_patients.csv",
    "test_labels_path": "dgit ata/ExtendedDAIC-lstm/test_patients.csv",
    "epochs": 60,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "weight_decay": 0.005,
    "lstm_hidden_size": 64,
    "lstm_layers": 2,
    "lstm_dropout": 0.4,  # Dropout between LSTM layers
    "classifier_dropout": 0.5,  # Dropout in the final classifier
    "input_features": 512,  # Feature dimension from CNN10
    "num_classes": 1,  # Binary classification (Depression/Not)
    "random_state": 42,
}


# --- 2. Model Architecture (LSTM with Attention) ---
class Attention(nn.Module):
    """A simple self-attention mechanism."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_scores = self.attention_weights_layer(lstm_output).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(
            1
        )
        return context_vector, attention_weights


class DepressionLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout_lstm,
        dropout_classifier,
    ):
        super(DepressionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_lstm if num_layers > 1 else 0,
        )
        self.attention = Attention(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        context_vector, _ = self.attention(lstm_out)
        output = self.classifier(context_vector)
        return output


# --- 3. Data Handling & Metrics ---
class PatientFeatureDataset(Dataset):
    """Dataset to load pre-computed feature tensors for each patient."""

    def __init__(self, feature_files, labels):
        self.feature_files = feature_files
        self.labels = labels

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        path = self.feature_files[idx]
        features = torch.load(path)
        label = self.labels[idx]
        return features, torch.tensor(label, dtype=torch.float32)


def collate_fn_pad(batch):
    """Pads sequences to the max length in a batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences, lengths, torch.stack(labels)


def get_metrics(targets, preds):
    """Calculate and return a dictionary of metrics, including UAR."""
    recall_per_class = recall_score(
        targets, preds, average=None, labels=[0, 1], zero_division=0
    )
    uar = np.mean(recall_per_class)

    return {
        "uar": uar,
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall_class_0": recall_per_class[0],
        "recall_class_1": recall_per_class[1],
    }


# --- 4. Main Training and Evaluation Script ---
def main():
    # Set seeds for reproducibility
    torch.manual_seed(config["random_state"])
    np.random.seed(config["random_state"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["random_state"])
        torch.cuda.manual_seed_all(config["random_state"])
        # Make CUDA operations deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create a meaningful run name with key hyperparameters and random seed
    run_name = f"lstm_h{config['lstm_hidden_size']}_l{config['lstm_layers']}_lr{config['learning_rate']}_bs{config['batch_size']}_wd{config['weight_decay']}_seed{config['random_state']}"

    # Initialize WANDB with run name and log all hyperparameters
    wandb.init(
        project=config["project_name"],
        config=config,
        name=run_name,
        tags=["lstm", "depression-detection", "attention"],
    )
    cfg = wandb.config

    # Log hyperparameters as a table for better visualization
    hyperparams_table = wandb.Table(columns=["Parameter", "Value"])
    for key, value in config.items():
        hyperparams_table.add_data(key, str(value))
    wandb.log({"hyperparameters": hyperparams_table})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run name: {run_name}")
    print(f"Random seed set to: {config['random_state']}")

    # --- Load and Prepare Data ---
    train_df = pd.read_csv(cfg.train_labels_path)
    val_df = pd.read_csv(cfg.val_labels_path)
    test_df = pd.read_csv(cfg.test_labels_path)

    # Construct full file paths
    train_df["feature_file_path"] = train_df["feature_file"].apply(
        lambda x: os.path.join(cfg.data_dir, x)
    )
    val_df["feature_file_path"] = val_df["feature_file"].apply(
        lambda x: os.path.join(cfg.data_dir, x)
    )
    test_df["feature_file_path"] = test_df["feature_file"].apply(
        lambda x: os.path.join(cfg.data_dir, x)
    )

    # Get file paths and labels
    train_paths = train_df["feature_file_path"].tolist()
    train_labels = train_df["PHQ_Binary"].tolist()

    val_paths = val_df["feature_file_path"].tolist()
    val_labels = val_df["PHQ_Binary"].tolist()

    test_paths = test_df["feature_file_path"].tolist()
    test_labels = test_df["PHQ_Binary"].tolist()

    train_dataset = PatientFeatureDataset(train_paths, train_labels)
    val_dataset = PatientFeatureDataset(val_paths, val_labels)
    test_dataset = PatientFeatureDataset(test_paths, test_labels)

    # Create generator for reproducible data loading
    generator = torch.Generator()
    generator.manual_seed(cfg.random_state)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_pad,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn_pad
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn_pad,
    )

    # --- Handle Class Imbalance with Weighted Loss ---
    num_neg = train_labels.count(0)
    num_pos = train_labels.count(1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    print(f"Class counts in training set: Negative={num_neg}, Positive={num_pos}")
    print(f"Positive class weight for loss function: {pos_weight.item():.2f}")

    # Log dataset statistics
    dataset_stats = {
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "train_neg_samples": num_neg,
        "train_pos_samples": num_pos,
        "train_class_ratio": num_pos / (num_neg + num_pos),
        "pos_weight": pos_weight.item(),
    }
    wandb.log(dataset_stats)

    # --- Initialize Model, Loss, and Optimizer ---
    model = DepressionLSTM(
        input_size=cfg.input_features,
        hidden_size=cfg.lstm_hidden_size,
        num_layers=cfg.lstm_layers,
        num_classes=cfg.num_classes,
        dropout_lstm=cfg.lstm_dropout,
        dropout_classifier=cfg.classifier_dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Log model architecture information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "optimizer": "AdamW",
        "loss_function": "BCEWithLogitsLoss",
    }
    wandb.log(model_info)
    print(
        f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)"
    )

    wandb.watch(model, criterion, log="all", log_freq=10)

    best_val_uar = -1.0

    # --- Training Loop ---
    for epoch in range(cfg.epochs):
        model.train()
        train_loss_total = 0.0

        for sequences, lengths, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item()

        # --- Validation Loop ---
        model.eval()
        val_loss_total = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for sequences, lengths, targets in val_loader:
                sequences, targets_device = sequences.to(device), targets.to(device)
                outputs = model(sequences, lengths)

                # Calculate validation loss
                loss = criterion(outputs.squeeze(), targets_device)
                val_loss_total += loss.item()

                # Get predictions for metrics
                preds = torch.sigmoid(outputs).squeeze().round().cpu().numpy()
                val_preds.extend(np.atleast_1d(preds).tolist())
                val_targets.extend(targets.cpu().numpy().tolist())

        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        val_metrics = get_metrics(val_targets, val_preds)

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val UAR: {val_metrics['uar']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

        # Log all metrics to WANDB
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_uar": val_metrics["uar"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall_class_0": val_metrics["recall_class_0"],
                "val_recall_class_1": val_metrics["recall_class_1"],
            }
        )

        # Save the best model based on validation UAR
        if val_metrics["uar"] > best_val_uar:
            best_val_uar = val_metrics["uar"]
            model_path = os.path.join(wandb.run.dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with Val UAR: {best_val_uar:.4f}")

    # --- Final Testing ---
    print(
        "\nTraining finished. Evaluating on the test set using the best model from validation."
    )
    best_model_path = os.path.join(wandb.run.dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for sequences, lengths, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences, lengths)
            preds = torch.sigmoid(outputs).squeeze().round().cpu().numpy()
            test_preds.extend(np.atleast_1d(preds).tolist())
            test_targets.extend(targets.cpu().numpy().tolist())

    test_metrics = get_metrics(test_targets, test_preds)
    print(
        f"\nTest Metrics:\n"
        f"UAR: {test_metrics['uar']:.4f}\n"
        f"Accuracy: {test_metrics['accuracy']:.4f}\n"
        f"F1-Score: {test_metrics['f1']:.4f}\n"
        f"Precision: {test_metrics['precision']:.4f}\n"
        f"Recall Class 0: {test_metrics['recall_class_0']:.4f}\n"
        f"Recall Class 1: {test_metrics['recall_class_1']:.4f}"
    )

    wandb.log(
        {
            "test_uar": test_metrics["uar"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall_class_0": test_metrics["recall_class_0"],
            "test_recall_class_1": test_metrics["recall_class_1"],
        }
    )

    # Generate and log confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Depressed", "Depressed"],
        yticklabels=["Non-Depressed", "Depressed"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Test Set Confusion Matrix")

    cm_path = os.path.join(wandb.run.dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    wandb.log({"confusion_matrix": wandb.Image(cm_path)})
    plt.close()

    wandb.finish()
    print("Script finished successfully.")


if __name__ == "__main__":
    main()
