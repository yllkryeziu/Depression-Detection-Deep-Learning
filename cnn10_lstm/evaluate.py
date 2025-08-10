from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix

from shared.common import FeatureSequenceDataset, pad_feature_batch
from .models import PatientLSTM


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg.paths.data_root)
    test_ds = FeatureSequenceDataset(root / "test.csv")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.eval.num_workers, collate_fn=pad_feature_batch)

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
    state = torch.load(Path(cfg.eval.ckpt_path) if cfg.eval.ckpt_path else Path(cfg.paths.runs_dir) / "patient_lstm_best.pt", map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for xb, lengths, yb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb, lengths).squeeze()
            preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy().tolist())
            targets.extend(yb.long().cpu().numpy().tolist())

    r = recall_score(targets, preds, average=None, labels=[0, 1], zero_division=0)
    metrics = {
        "uar": float(np.mean(r)),
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall_class_0": r[0],
        "recall_class_1": r[1],
        "confusion": confusion_matrix(targets, preds).tolist(),
    }
    print(metrics)


if __name__ == "__main__":
    main()


