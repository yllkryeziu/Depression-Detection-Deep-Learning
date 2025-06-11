import os
import shutil
from typing import Dict, List, Optional, Union
import requests
import tarfile
from functools import cached_property

from omegaconf import DictConfig
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from autrainer.datasets import BaseClassificationDataset

"""
Load the dataset base url from the environment variables.
You should request access to the dataset from the authors
"""

load_dotenv()
BASE_URL = os.getenv("BASE_URL")

DEFAULT_SR_TARGET = 16000  # Sample rate for audio
DEFAULT_BUFFER_SEC = 0.25  # Buffer in seconds
DEFAULT_MAX_UTT_SEC = 90.0  # Maximum utterance length in seconds
DEFAULT_SPLIT_UTTERANCES = True 
DEFAULT_MERGE_UTTERANCES = True
DEFAULT_PATIENT_IDS = list(range(363, 719))  # Full range [300, 719)

class ExtendedDAIC(BaseClassificationDataset):
    """
    E-DAIC dataset for depression detection.

    download():
      • Fetch train/dev/test CSVs
      • Download & extract each <pid>_P.tar.gz with progress visualization
      • Prune to keep only <pid>_AUDIO.wav and <pid>_Transcript.csv, delete archive

    preprocess():
      • Optionally split utterances (slice by transcript timestamps)
      • Optionally merge utterances back into one WAV
    """

    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: list,
        tracking_metric,
        index_column: str,
        target_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        features_path: Optional[str] = None,
        split_utterances: bool = DEFAULT_SPLIT_UTTERANCES,
        merge_utterances: bool = DEFAULT_MERGE_UTTERANCES,
        sr_target: int = DEFAULT_SR_TARGET,
        buffer_sec: float = DEFAULT_BUFFER_SEC,
        max_utt_sec: float = DEFAULT_MAX_UTT_SEC,
        patient_range: Optional[List[int]] = None,
        chunk_dur: float = 0,
        train_transform=None,
        dev_transform=None,
        test_transform=None,
        **kwargs
    ) -> None:
        self.split_utterances = split_utterances
        self.merge_utterances = merge_utterances
        self.sr_target = sr_target
        self.buffer_sec = buffer_sec
        self.max_utt_sec = max_utt_sec
        self.chunk_dur = chunk_dur
        
        if patient_range:
            self.patient_ids = list(range(patient_range[0], patient_range[1]))
        else:
            patients_dir = Path(path) / "patients"
            if patients_dir.exists():
                patient_dirs = [d for d in patients_dir.iterdir() if d.is_dir() and d.name.endswith('_P')]
                self.patient_ids = sorted([int(d.name.replace('_P', '')) for d in patient_dirs])
            else:
                self.patient_ids = DEFAULT_PATIENT_IDS
        
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            features_path=features_path,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            **kwargs
        )
        
    @staticmethod
    def download(path: str) -> None:
        p = Path(path)
        # 1) Download split CSVs directly to dataset path
        if BASE_URL is None:
            raise ValueError("BASE_URL environment variable not set. Please ensure it is in your .env file or environment.")

        for split in ["train_split.csv", "dev_split.csv", "test_split.csv"]:
            out_csv = os.path.join(p, split)
            if not os.path.exists(out_csv):
                url = os.path.join(BASE_URL, "labels", split)
                print(f"Downloading {split} from {url}...")
                try:
                    r = requests.get(url, stream=True, timeout=30)
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    block_size = 8192
                    with open(out_csv, "wb") as f, tqdm(
                        total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {split}"
                    ) as bar:
                        for chunk in r.iter_content(block_size):
                            f.write(chunk)
                            bar.update(len(chunk))
                    if total_size != 0 and bar.n != total_size:
                        print(f"Error: Downloaded size of {split} does not match expected size.")
                    else:
                        print(f"Successfully downloaded {out_csv}")

                    # Rename the downloaded file
                    new_csv_name = split.replace('_split', '')
                    new_csv_path = os.path.join(p, new_csv_name)
                    try:
                        os.rename(out_csv, new_csv_path)
                        print(f"Renamed {out_csv} to {new_csv_path}")
                    except OSError as e:
                        print(f"Error renaming {out_csv} to {new_csv_path}: {e}")

                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {split}: {e}")
                    if os.path.exists(out_csv):
                        os.remove(out_csv)
                    continue

        # 2) Prepare folders
        patients_dir = os.path.join(p, "patients")
        os.makedirs(patients_dir, exist_ok=True)
        interim_dir = os.path.join(p, "interim")
        os.makedirs(interim_dir, exist_ok=True)
        merged_dir = os.path.join(p, "default")
        os.makedirs(merged_dir, exist_ok=True)

        # 3) Per patient: Download, extract, slice, and merge immediately
        for pid in DEFAULT_PATIENT_IDS:
            archive_name = f"{pid}_P.tar.gz"
            archive_url = os.path.join(BASE_URL, "data", archive_name)
            local_archive_path = os.path.join(patients_dir, archive_name)
            patient_extract_dir = os.path.join(patients_dir, f"{pid}_P")

            # Check if final merged output already exists
            final_merged_file = os.path.join(merged_dir, str(pid), f"{pid}_P.wav")
            if os.path.exists(final_merged_file):
                print(f"Patient {pid} final merged audio already exists at {final_merged_file}. Skipping.")
                continue
            
            if os.path.exists(local_archive_path):
                 print(f"Archive {local_archive_path} exists but final output not found. Removing archive to redownload.")
                 os.remove(local_archive_path)

            print(f"Processing patient {pid}: Downloading {archive_name} from {archive_url}...")
            try:
                # Download archive
                r = requests.get(archive_url, stream=True, timeout=600)
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 8192
                with open(local_archive_path, "wb") as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {archive_name}"
                ) as bar:
                    for chunk in r.iter_content(block_size):
                        f.write(chunk)
                        bar.update(len(chunk))
                
                if total_size != 0 and bar.n != total_size:
                    print(f"Error: Downloaded size of {archive_name} does not match expected size.")
                    if os.path.exists(local_archive_path): os.remove(local_archive_path)
                    continue

                # Extract archive
                with tarfile.open(local_archive_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        tar.extract(member, path=patients_dir)
                
                # Remove features directory if it exists
                features_dir = os.path.join(patient_extract_dir, "features")
                if os.path.exists(features_dir) and os.path.isdir(features_dir):
                    shutil.rmtree(features_dir)
                
                print(f"Extraction complete for patient {pid}")

                # Immediately slice and merge for this patient
                print(f"Slicing and merging patient {pid}...")
                
                # Slice utterances
                pf = patient_extract_dir
                if not os.path.exists(pf):
                    print(f"Patient directory {pf} not found after extraction, skipping...")
                    continue
                    
                # Find transcript and columns
                try:
                    tx = next(f for f in os.listdir(pf) if "Transcript" in f)
                    df = pd.read_csv(os.path.join(pf, tx))
                    df.columns = [c.lower() for c in df.columns]
                    start_c = next(c for c in df.columns if "start" in c)
                    end_c   = next(c for c in df.columns if "end"   in c)
                except (StopIteration, FileNotFoundError) as e:
                    print(f"Error processing transcript for patient {pid}: {e}")
                    continue

                # Read + resample interview
                try:
                    wav0 = next(f for f in os.listdir(pf) if f.lower().endswith(".wav"))
                    y, orig_sr = sf.read(os.path.join(pf, wav0))
                    if orig_sr != DEFAULT_SR_TARGET:
                        y = librosa.resample(y, orig_sr=orig_sr, target_sr=DEFAULT_SR_TARGET)
                    total = len(y)
                except (StopIteration, FileNotFoundError, sf.SoundFileError) as e:
                    print(f"Error processing audio for patient {pid}: {e}")
                    continue

                # Create interim directory for this patient
                out_dir = os.path.join(interim_dir, str(pid))
                os.makedirs(out_dir, exist_ok=True)

                # Slice utterances
                for i, row in df.iterrows():
                    s = int(row[start_c] * DEFAULT_SR_TARGET)
                    e = int(min(total, (row[end_c] + DEFAULT_BUFFER_SEC) * DEFAULT_SR_TARGET))
                    if (e - s) > DEFAULT_MAX_UTT_SEC * DEFAULT_SR_TARGET:
                        continue
                    sf.write(
                        os.path.join(out_dir, f"{pid}_{i:04d}.wav"),
                        y[s:e],
                        DEFAULT_SR_TARGET
                    )

                # Merge utterances
                utt_dir = out_dir
                if os.path.isdir(utt_dir):
                    parts = sorted(
                        f for f in os.listdir(utt_dir)
                        if f.lower().endswith(".wav")
                    )
                    if parts:
                        try:
                            audio = np.concatenate([
                                sf.read(os.path.join(utt_dir, f))[0] for f in parts
                            ], axis=0)

                            subj_dir = os.path.join(merged_dir, str(pid))
                            os.makedirs(subj_dir, exist_ok=True)
                            out_wav = os.path.join(subj_dir, f"{pid}_P.wav")
                            sf.write(out_wav, audio, DEFAULT_SR_TARGET)

                            # Cleanup interim utterances
                            shutil.rmtree(utt_dir, ignore_errors=True)
                            print(f"Successfully processed and merged audio for patient {pid}")
                        except Exception as e:
                            print(f"Error processing merged audio for patient {pid}: {e}")
                            continue
                    else:
                        print(f"No utterance files found for patient {pid}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading/processing {archive_name} for patient {pid}: {e}")
            except tarfile.TarError as e:
                print(f"Error extracting {local_archive_path}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred for patient {pid}: {e}")
            finally:
                if os.path.exists(local_archive_path):
                   os.remove(local_archive_path)
        
        if os.path.exists(interim_dir) and not os.listdir(interim_dir):
            os.rmdir(interim_dir)
        
        print("All patient data processing completed.")

    @cached_property
    def df_train(self):
        return self._build_split("train.csv")

    @cached_property
    def df_dev(self):
        return self._build_split("dev.csv")

    @cached_property
    def df_test(self):
        return self._build_split("test.csv")

    @property
    def output_dim(self):
        # binary PHQ_Binary
        return 2

# ExtendedDAIC specific methods

    def _build_split(self, split_file: str):
        df = pd.read_csv(os.path.join(self.path, split_file))
        
        def get_relative_path(pid):
            # Use the file_type to determine the correct extension
            file_name = f"{pid}_P.npy"
            wav_f = os.path.join(str(pid), file_name)
            full_path = os.path.join(self.path, self.features_subdir, wav_f)
            if os.path.isfile(full_path):
                return wav_f
            return None
        
        df['file_path'] = df['Participant_ID'].apply(get_relative_path)
        df = df.dropna(subset=['file_path'])
        
        df['Participant_ID'] = df['file_path']

        df = df.drop('file_path', axis=1)
        
        return df