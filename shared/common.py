from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MelDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"])  # (mel, time)
        x = torch.from_numpy(mel).unsqueeze(0)
        y = torch.tensor(float(row["PHQ_Binary"]))
        pid = int(row["Participant_ID"])
        return x, y, pid


def create_segment_loaders(root: Path, batch_size: int, num_workers: int):
    train_csv = root / "train.csv"
    dev_csv = root / "dev.csv"
    test_csv = root / "test.csv"

    train_ds = MelDataset(train_csv)
    dev_ds = MelDataset(dev_csv)
    test_ds = MelDataset(test_csv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, dev_loader, test_loader


class FeatureSequenceDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)
        self.groups = self.df.groupby("Participant_ID")
        self.patient_ids = list(self.groups.groups.keys())

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        gdf = self.groups.get_group(pid).sort_values(["segment_index"])  # temporal order
        feats = [torch.load(p) for p in gdf["feature_path"].tolist()]
        x = torch.stack(feats, dim=0)
        y = torch.tensor(float(gdf["PHQ_Binary"].iloc[0]))
        return x, y, int(pid)


def pad_feature_batch(batch):
    sequences, labels, pids = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    feat_dim = sequences[0].shape[1]
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(sequences), max_len, feat_dim), dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        padded[i, : seq.shape[0]] = seq
    return padded, lengths, torch.stack(labels), torch.tensor(pids)


