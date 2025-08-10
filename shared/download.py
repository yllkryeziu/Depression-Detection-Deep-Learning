import os
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm
import hydra


def _download_file(url: str, local_path: Path, description: str) -> bool:
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, stream=True, timeout=600)
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        block_size = 8192
        with open(local_path, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True, desc=description) as bar:
            for chunk in r.iter_content(block_size):
                f.write(chunk)
                bar.update(len(chunk))
        if total_size != 0 and bar.n != total_size:
            if local_path.exists():
                local_path.unlink(missing_ok=True)
            return False
        return True
    except requests.exceptions.RequestException:
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        return False


def _load_patient_ids(ids_file: Path, default_ranges: List[List[int]]) -> List[int]:
    if ids_file.exists():
        ids: List[int] = []
        for line in ids_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            ids.append(int(line))
        return ids
    ids: List[int] = []
    for r in default_ranges:
        ids.extend(list(range(r[0], r[1])))
    return ids


def _get_patient_row(pid: int, split_csvs: Dict[str, pd.DataFrame]) -> pd.Series | None:
    for _, df in split_csvs.items():
        row = df[df["Participant_ID"] == pid]
        if not row.empty:
            return row.iloc[0]
    return None


def _create_utterance_row(filename: str, pid: int, patient_info: pd.Series) -> Dict:
    phq_binary = 1 if patient_info["PHQ_Score"] >= 10 else 0
    return {
        "filename": filename,
        "Participant_ID": pid,
        "Gender": patient_info.get("Gender"),
        "PHQ_Binary": phq_binary,
        "PHQ_Score": patient_info.get("PHQ_Score"),
        "PCL-C (PTSD)": patient_info.get("PCL-C (PTSD)"),
        "PTSD Severity": patient_info.get("PTSD Severity"),
    }


@hydra.main(config_path="../cnn10_baseline/conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    base_url = cfg.dataset.base_url
    if base_url is None:
        raise ValueError("BASE_URL not provided. Set env BASE_URL or dataset.base_url in config.")

    data_root = Path(cfg.paths.raw_dir)
    patients_dir = Path(cfg.paths.patients_dir)
    transcripts_dir = Path(cfg.paths.raw_dir) / "transcripts"
    default_dir = Path(cfg.paths.data_root) / "default"
    data_root.mkdir(parents=True, exist_ok=True)
    patients_dir.mkdir(parents=True, exist_ok=True)
    default_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    split_csvs: Dict[str, pd.DataFrame] = {}
    for split in ["train_split.csv", "dev_split.csv", "test_split.csv"]:
        tmp_csv = data_root / f"temp_{split}"
        if not tmp_csv.exists():
            url = f"{base_url}/labels/{split}"
            _download_file(url, tmp_csv, f"Downloading {split}")
        split_csvs[split] = pd.read_csv(tmp_csv)

    patient_ids = _load_patient_ids(Path(cfg.dataset.patients.from_file), cfg.dataset.patients.default_ranges)
    utterance_rows: List[Dict] = []

    for pid in patient_ids:
        archive = patients_dir / f"{pid}_P.tar.gz"
        extract_dir = patients_dir / f"{pid}_P"
        existing = [f for f in default_dir.glob(f"{pid}_*.wav")]
        if existing:
            row = _get_patient_row(pid, split_csvs)
            if row is not None:
                for wav in existing:
                    utterance_rows.append(_create_utterance_row(wav.name, pid, row))
            continue
        if archive.exists():
            archive.unlink()
        url = f"{base_url}/data/{pid}_P.tar.gz"
        ok = _download_file(url, archive, f"Downloading {pid}_P.tar.gz")
        if not ok:
            continue
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=patients_dir)
            features_dir = extract_dir / "features"
            if features_dir.exists() and features_dir.is_dir():
                shutil.rmtree(features_dir)
            tx_files = [f for f in os.listdir(extract_dir) if "Transcript" in f]
            wav_files = [f for f in os.listdir(extract_dir) if f.lower().endswith(".wav")]
            if not tx_files or not wav_files:
                continue
            row = _get_patient_row(pid, split_csvs)
            if row is None:
                continue
            wav0 = wav_files[0]
            dest = default_dir / f"{pid}_AUDIO.wav"
            shutil.copy2(extract_dir / wav0, dest)
            tx0 = tx_files[0]
            tx_dest = transcripts_dir / f"{pid}_Transcript.csv"
            shutil.copy2(extract_dir / tx0, tx_dest)
            utterance_rows.append(_create_utterance_row(dest.name, pid, row))
        except tarfile.TarError:
            pass
        finally:
            if archive.exists():
                archive.unlink()
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

    utterance_df = pd.DataFrame(utterance_rows)
    if not utterance_df.empty:
        for split_file, split_df in split_csvs.items():
            split_name = split_file.replace("_split.csv", ".csv")
            split_patients = set(split_df["Participant_ID"].tolist())
            split_subset = utterance_df[utterance_df["Participant_ID"].isin(split_patients)]
            if not split_subset.empty:
                out_csv = Path(cfg.paths.data_root) / split_name
                split_subset.to_csv(out_csv, index=False)

    for split in ["train_split.csv", "dev_split.csv", "test_split.csv"]:
        tmp_csv = data_root / f"temp_{split}"
        if tmp_csv.exists():
            tmp_csv.unlink()

    print("Download step finished.")


if __name__ == "__main__":
    main()


