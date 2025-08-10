from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from cnn10_baseline.models import CNN10


@hydra.main(config_path="../cnn10_baseline/conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_dir = Path(cfg.paths.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    backbone = CNN10(num_classes=cfg.model.num_classes, dropout=cfg.model.dropout)
    ckpt_path = Path(cfg.eval.ckpt_path) if cfg.eval.ckpt_path else Path(cfg.paths.runs_dir) / "cnn10_best.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state)
        backbone.load_state_dict(sd, strict=False)
    feature_extractor = nn.Sequential(*list(backbone.net.children()))
    head = backbone.fc
    feature_extractor.to(device).eval()

    for split in ["train.csv", "dev.csv", "test.csv"]:
        csv_path = Path(cfg.paths.data_root) / split
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        out_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"extract {split}"):
            mel = np.load(row["mel_path"])  # (mel, time)
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = feature_extractor(x)  # (1, 512, 1, 1)
                feat = feat.flatten(1)      # (1, 512)
            feat_path = features_dir / (Path(row["mel_path"]).stem + ".pt")
            torch.save(feat.cpu(), feat_path)
            out = dict(row)
            out.update({"feature_path": str(feat_path)})
            out_rows.append(out)
        pd.DataFrame(out_rows).to_csv(csv_path, index=False)

    print("Feature extraction finished.")


if __name__ == "__main__":
    main()


