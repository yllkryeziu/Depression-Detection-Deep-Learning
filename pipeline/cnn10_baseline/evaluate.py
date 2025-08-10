from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix

from ..data.common import create_segment_loaders
from .models import CNN10


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = create_segment_loaders(Path(cfg.paths.data_root), cfg.eval.batch_size, cfg.eval.num_workers)
    state = torch.load(Path(cfg.eval.ckpt_path) if cfg.eval.ckpt_path else Path(cfg.paths.runs_dir) / "cnn10_best.pt", map_location=device)
    model = CNN10(num_classes=cfg.model.num_classes, dropout=cfg.model.dropout).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb).squeeze()
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


