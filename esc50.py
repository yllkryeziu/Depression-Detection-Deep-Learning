import os
import shutil
from typing import Dict, List, Tuple

import pandas as pd
from autrainer.datasets import BaseClassificationDataset
from autrainer.datasets.utils import ZipDownloadManager
from autrainer.transforms import SmartCompose
from omegaconf import DictConfig

FILES = {
    "ESC-50-master.zip": "https://github.com/karoldvl/ESC-50/archive/master.zip"
}


class ESC50(BaseClassificationDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[str | DictConfig | Dict],
        tracking_metric: str | DictConfig | Dict,
        index_column: str,
        target_column: str | List[str],
        file_type: str,
        file_handler: str | DictConfig | Dict,
        batch_size: int,
        inference_batch_size: int | None = None,
        train_transform: SmartCompose | None = None,
        dev_transform: SmartCompose | None = None,
        test_transform: SmartCompose | None = None,
        stratify: List[str] | None = None,
        train_folds: List[int] | None = None,
        dev_folds: List[int] | None = None,
        test_folds: List[int] | None = None,
    ) -> None:
        self.train_folds = train_folds or [1, 2, 3]
        self.dev_folds = dev_folds or [4]
        self.test_folds = test_folds or [5]
        super().__init__(
            path,
            features_subdir,
            seed,
            metrics,
            tracking_metric,
            index_column,
            target_column,
            file_type,
            file_handler,
            batch_size,
            inference_batch_size,
            train_transform,
            dev_transform,
            test_transform,
            stratify,
        )

    def load_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        meta = pd.read_csv(os.path.join(self.path, "esc50.csv"))
        return (
            meta[meta["fold"].isin(self.train_folds)],
            meta[meta["fold"].isin(self.dev_folds)],
            meta[meta["fold"].isin(self.test_folds)],
        )

    def download(path: str) -> None:
        out_path = os.path.join(path, "default")
        if os.path.isdir(out_path):
            return

        os.makedirs(path, exist_ok=True)
        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(check_exist=["ESC-50-master.zip"])
        dl_manager.extract(check_exist=["ESC-50-master"])

        shutil.move(os.path.join(path, "ESC-50-master", "audio"), out_path)
        shutil.move(
            os.path.join(path, "ESC-50-master", "meta", "esc50.csv"),
            path,
        )

        shutil.rmtree(os.path.join(path, "ESC-50-master"))
