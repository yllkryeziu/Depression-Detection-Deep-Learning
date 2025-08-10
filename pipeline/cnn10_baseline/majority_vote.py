from pathlib import Path
import pandas as pd
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from collections import defaultdict
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score

from ..shared.common import create_segment_loaders
from .models import CNN10


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = create_segment_loaders(Path(cfg.paths.data_root), cfg.eval.batch_size, cfg.eval.num_workers)

    ckpt_path = Path(cfg.eval.ckpt_path) if cfg.eval.ckpt_path else Path(cfg.paths.runs_dir) / "cnn10_best.pt"
    state = torch.load(ckpt_path, map_location=device)
    model = CNN10(num_classes=cfg.model.num_classes, dropout=cfg.model.dropout).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    patient_votes = defaultdict(list)
    patient_labels = {}

    test_csv = Path(cfg.paths.data_root) / "test.csv"
    meta_df = pd.read_csv(test_csv)

    with torch.no_grad():
        for i, (xb, yb, _) in enumerate(test_loader):
            xb = xb.to(device)
            logits = model(xb).squeeze()
            pred = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            batch_idx_start = i * cfg.eval.batch_size
            batch_idx_end = batch_idx_start + len(yb)
            rows = meta_df.iloc[batch_idx_start:batch_idx_end]
            for pid, p, y in zip(rows["Participant_ID"].tolist(), pred.tolist(), yb.tolist()):
                patient_votes[int(pid)].append(int(p))
                patient_labels[int(pid)] = int(y)

    preds, targets = [], []
    for pid, votes in patient_votes.items():
        vote = int(np.round(np.mean(votes)))
        preds.append(vote)
        targets.append(patient_labels[pid])

    r = recall_score(targets, preds, average=None, labels=[0, 1], zero_division=0)
    metrics = {
        "uar": float(np.mean(r)),
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall_class_0": r[0],
        "recall_class_1": r[1],
    }
    print(metrics)


if __name__ == "__main__":
    main()


