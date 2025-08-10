from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import wandb
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score

from ..data.common import FeatureSequenceDataset, pad_feature_batch
from .models import PatientLSTM


def compute_metrics(targets, preds):
    r = recall_score(targets, preds, average=None, labels=[0, 1], zero_division=0)
    return {
        "uar": float(np.mean(r)),
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall_class_0": r[0],
        "recall_class_1": r[1],
    }


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg, tags=list(cfg.wandb.tags))

    root = Path(cfg.paths.data_root)
    train_ds = FeatureSequenceDataset(root / "train.csv")
    dev_ds = FeatureSequenceDataset(root / "dev.csv")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, collate_fn=pad_feature_batch)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, collate_fn=pad_feature_batch)

    r = cfg.model.rnn
    model = PatientLSTM(
        input_dim=cfg.model.segment_feature_dim,
        hidden_size=r.hidden_size,
        num_layers=r.num_layers,
        dropout_lstm=r.dropout,
        bidirectional=r.bidirectional,
        classifier_dropout=cfg.model.classifier_dropout,
        num_classes=cfg.model.num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    y_train = [int(y.item()) for _, y, _ in train_ds]
    num_neg = y_train.count(0)
    num_pos = y_train.count(1)
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32, device=device) if cfg.train.pos_weight_auto else torch.tensor([1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -1e9
    patience = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        for xb, lengths, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lengths).squeeze()
            loss = criterion(logits, yb)
            loss.backward()
            if cfg.train.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for xb, lengths, yb, _ in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb, lengths).squeeze()
                val_loss += criterion(logits, yb).item()
                preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy().tolist())
                targets.extend(yb.long().cpu().numpy().tolist())

        log = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, len(train_loader)),
            "val_loss": val_loss / max(1, len(dev_loader)),
        }
        m = compute_metrics(targets, preds)
        log.update({f"val_{k}": v for k, v in m.items()})
        if cfg.wandb.enabled:
            wandb.log(log)

        if m["uar"] > best_val:
            best_val = m["uar"]
            patience = 0
            ckpt = Path(cfg.paths.runs_dir) / "patient_lstm_best.pt"
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt)
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                break

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()


