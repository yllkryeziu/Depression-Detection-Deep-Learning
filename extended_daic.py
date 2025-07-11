import os
import shutil
from typing import Dict, List, Optional, Union
import requests
import tarfile

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
DEFAULT_PATIENT_IDS = (
    list(range(300, 493)) + list(range(600, 719)) + list(range(600, 719))
)  # Exclude 493-599 as they don't exist
DEFAULT_SEGMENT_DURATION = 6.0  # 6 seconds
DEFAULT_OVERLAP_RATIO = 0.33  # 50% overlap


def create_overlapping_segments(
    merged_audio: np.ndarray,
    sr: int,
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> List[np.ndarray]:
    """
    Create overlapping segments from merged audio.

    Args:
        merged_audio: The merged audio array
        sr: Sample rate
        segment_duration: Duration of each segment in seconds
        overlap_ratio: Overlap ratio (0.5 = 50% overlap)

    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sr)
    hop_samples = int(segment_samples * (1 - overlap_ratio))

    segments = []
    start = 0

    while start + segment_samples <= len(merged_audio):
        segment = merged_audio[start : start + segment_samples]
        segments.append(segment)
        start += hop_samples

    # Add the last segment if there's remaining audio
    if start < len(merged_audio):
        remaining = merged_audio[start:]
        if (
            len(remaining) > segment_samples * 0.5
        ):  # Only add if segment is at least 50% of target length
            # Pad the last segment to match the target length
            padded_segment = np.pad(
                remaining,
                (0, segment_samples - len(remaining)),
                mode="constant",
                constant_values=0,
            )
            segments.append(padded_segment)

    return segments


class ExtendedDAIC(BaseClassificationDataset):
    """
    E-DAIC dataset for depression detection.

    download():
      • Fetch train/dev/test CSVs
      • Download & extract each <pid>_P.tar.gz with progress visualization
      • Prune to keep only <pid>_AUDIO.wav and <pid>_Transcript.csv, delete archive

    preprocess():
      • Split utterances (slice by transcript timestamps) and save as individual files
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
        sr_target: int = DEFAULT_SR_TARGET,
        buffer_sec: float = DEFAULT_BUFFER_SEC,
        max_utt_sec: float = DEFAULT_MAX_UTT_SEC,
        patient_range: Optional[List[int]] = None,
        chunk_dur: float = 0,
        train_transform=None,
        dev_transform=None,
        test_transform=None,
        **kwargs,
    ) -> None:
        self.split_utterances = split_utterances
        self.sr_target = sr_target
        self.buffer_sec = buffer_sec
        self.max_utt_sec = max_utt_sec
        self.chunk_dur = chunk_dur

        if patient_range:
            self.patient_ids = list(range(patient_range[0], patient_range[1]))
        else:
            patients_dir = Path(path) / "patients"
            if patients_dir.exists():
                patient_dirs = [
                    d
                    for d in patients_dir.iterdir()
                    if d.is_dir() and d.name.endswith("_P")
                ]
                self.patient_ids = sorted(
                    [int(d.name.replace("_P", "")) for d in patient_dirs]
                )
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
            **kwargs,
        )

    @staticmethod
    def download(path: str) -> None:
        p = Path(path)
        # Ensure the dataset directory exists
        p.mkdir(parents=True, exist_ok=True)

        # 1) Download split CSVs directly to dataset path
        if BASE_URL is None:
            raise ValueError(
                "BASE_URL environment variable not set. Please ensure it is in your .env file or environment."
            )

        # Download original patient-based CSV files to a temporary location
        temp_csvs = {}
        for split in ["train_split.csv", "dev_split.csv", "test_split.csv"]:
            temp_csv = os.path.join(p, f"temp_{split}")
            if not os.path.exists(temp_csv):
                url = os.path.join(BASE_URL, "labels", split)
                print(f"Downloading {split} from {url}...")
                try:
                    r = requests.get(url, stream=True, timeout=30)
                    r.raise_for_status()
                    total_size = int(r.headers.get("content-length", 0))
                    block_size = 8192
                    with (
                        open(temp_csv, "wb") as f,
                        tqdm(
                            total=total_size,
                            unit="iB",
                            unit_scale=True,
                            desc=f"Downloading {split}",
                        ) as bar,
                    ):
                        for chunk in r.iter_content(block_size):
                            f.write(chunk)
                            bar.update(len(chunk))
                    if total_size != 0 and bar.n != total_size:
                        print(
                            f"Error: Downloaded size of {split} does not match expected size."
                        )
                    else:
                        print(f"Successfully downloaded {temp_csv}")

                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {split}: {e}")
                    if os.path.exists(temp_csv):
                        os.remove(temp_csv)
                    continue

            # Read the CSV to store patient information
            temp_csvs[split] = pd.read_csv(temp_csv)

        # 2) Prepare folders
        patients_dir = os.path.join(p, "patients")
        os.makedirs(patients_dir, exist_ok=True)
        default_dir = os.path.join(p, "default")
        os.makedirs(default_dir, exist_ok=True)

        # 3) Per patient: Download, extract, and slice immediately - save to flat structure
        all_utterance_data = []  # Store information about all utterances

        for pid in DEFAULT_PATIENT_IDS:
            archive_name = f"{pid}_P.tar.gz"
            archive_url = os.path.join(BASE_URL, "data", archive_name)
            local_archive_path = os.path.join(patients_dir, archive_name)
            patient_extract_dir = os.path.join(patients_dir, f"{pid}_P")

            # Check if utterances for this patient already exist in flat structure
            existing_utterances = [
                f
                for f in os.listdir(default_dir)
                if f.startswith(f"{pid}_") and f.endswith(".wav")
            ]
            if existing_utterances:
                print(
                    f"Patient {pid} utterances already exist in flat structure. Skipping download."
                )
                # Still need to collect the utterance information for CSV generation
                patient_info = None
                for split_name, split_df in temp_csvs.items():
                    patient_row = split_df[split_df["Participant_ID"] == pid]
                    if not patient_row.empty:
                        patient_info = patient_row.iloc[0]
                        break

                if patient_info is not None:
                    for utterance_file in existing_utterances:
                        utterance_data = {
                            "filename": utterance_file,
                            "Participant_ID": pid,
                            "Gender": patient_info["Gender"],
                            "PHQ_Binary": patient_info["PHQ_Binary"],
                            "PHQ_Score": patient_info["PHQ_Score"],
                            "PCL-C (PTSD)": patient_info["PCL-C (PTSD)"],
                            "PTSD Severity": patient_info["PTSD Severity"],
                        }
                        all_utterance_data.append(utterance_data)
                continue

            # Check if WAV and transcript files already exist in patients directory
            if os.path.exists(patient_extract_dir):
                wav_files = [
                    f
                    for f in os.listdir(patient_extract_dir)
                    if f.lower().endswith(".wav")
                ]
                transcript_files = [
                    f for f in os.listdir(patient_extract_dir) if "Transcript" in f
                ]

                if wav_files and transcript_files:
                    print(
                        f"Patient {pid} WAV and transcript already exist in {patient_extract_dir}. Processing existing files..."
                    )
                    # Skip download and extraction, go directly to slicing
                    try:
                        # Get patient information from CSV
                        patient_info = None
                        for split_name, split_df in temp_csvs.items():
                            patient_row = split_df[split_df["Participant_ID"] == pid]
                            if not patient_row.empty:
                                patient_info = patient_row.iloc[0]
                                break

                        if patient_info is None:
                            print(
                                f"No patient information found for {pid}, skipping..."
                            )
                            continue

                        # Immediately slice utterances for this patient
                        print(f"Slicing utterances for patient {pid}...")

                        # Find transcript and columns
                        tx = transcript_files[0]  # Use the first transcript file found
                        df = pd.read_csv(os.path.join(patient_extract_dir, tx))
                        df.columns = [c.lower() for c in df.columns]
                        start_c = next(c for c in df.columns if "start" in c)
                        end_c = next(c for c in df.columns if "end" in c)

                        # Read audio and check sample rate
                        wav0 = wav_files[0]  # Use the first WAV file found
                        y, orig_sr = sf.read(os.path.join(patient_extract_dir, wav0))
                        # Skip 48kHz audio for now
                        if orig_sr == 48000:
                            print(f"Skipping patient {pid}: audio is 48kHz")
                            continue
                        # Resample if needed (but not 48kHz)
                        if orig_sr != DEFAULT_SR_TARGET:
                            y = librosa.resample(
                                y, orig_sr=orig_sr, target_sr=DEFAULT_SR_TARGET
                            )
                        total = len(y)

                        # First, extract all utterances
                        utterances = []
                        for i, row in df.iterrows():
                            s = int(row[start_c] * DEFAULT_SR_TARGET)
                            e = int(
                                min(
                                    total,
                                    (row[end_c] + DEFAULT_BUFFER_SEC)
                                    * DEFAULT_SR_TARGET,
                                )
                            )
                            if (e - s) > DEFAULT_MAX_UTT_SEC * DEFAULT_SR_TARGET:
                                continue
                            utterances.append(y[s:e])

                        # Merge all utterances back into a single audio stream
                        if utterances:
                            merged_audio = np.concatenate(utterances)

                            # Create 6-second segments with 50% overlap
                            segments = create_overlapping_segments(
                                merged_audio, DEFAULT_SR_TARGET
                            )

                            # Save each segment as a separate file
                            utterance_count = 0
                            for i, segment in enumerate(segments):
                                utterance_filename = f"{pid}_{i:04d}.wav"
                                utterance_path = os.path.join(
                                    default_dir, utterance_filename
                                )
                                sf.write(utterance_path, segment, DEFAULT_SR_TARGET)

                                # Store utterance information
                                utterance_data = {
                                    "filename": utterance_filename,
                                    "Participant_ID": pid,
                                    "Gender": patient_info["Gender"],
                                    "PHQ_Binary": patient_info["PHQ_Binary"],
                                    "PHQ_Score": patient_info["PHQ_Score"],
                                    "PCL-C (PTSD)": patient_info["PCL-C (PTSD)"],
                                    "PTSD Severity": patient_info["PTSD Severity"],
                                }
                                all_utterance_data.append(utterance_data)
                                utterance_count += 1

                        else:
                            utterance_count = 0
                            print(f"No valid utterances found for patient {pid}")

                        print(
                            f"Successfully processed {utterance_count} segments for patient {pid}"
                        )
                        continue  # Skip to next patient

                    except Exception as e:
                        print(f"Error processing existing files for patient {pid}: {e}")
                        print("Will attempt to re-download and extract...")
                        # If processing existing files fails, continue with download

            if os.path.exists(local_archive_path):
                print(
                    f"Archive {local_archive_path} exists but utterances not found. Removing archive to redownload."
                )
                os.remove(local_archive_path)

            print(
                f"Processing patient {pid}: Downloading {archive_name} from {archive_url}..."
            )
            try:
                # Download archive
                r = requests.get(archive_url, stream=True, timeout=600)
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                block_size = 8192
                with (
                    open(local_archive_path, "wb") as f,
                    tqdm(
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        desc=f"Downloading {archive_name}",
                    ) as bar,
                ):
                    for chunk in r.iter_content(block_size):
                        f.write(chunk)
                        bar.update(len(chunk))

                if total_size != 0 and bar.n != total_size:
                    print(
                        f"Error: Downloaded size of {archive_name} does not match expected size."
                    )
                    if os.path.exists(local_archive_path):
                        os.remove(local_archive_path)
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

                # Get patient information from CSV
                patient_info = None
                for split_name, split_df in temp_csvs.items():
                    patient_row = split_df[split_df["Participant_ID"] == pid]
                    if not patient_row.empty:
                        patient_info = patient_row.iloc[0]
                        break

                if patient_info is None:
                    print(f"No patient information found for {pid}, skipping...")
                    continue

                # Immediately slice utterances for this patient
                print(f"Slicing utterances for patient {pid}...")

                # Slice utterances
                pf = patient_extract_dir
                if not os.path.exists(pf):
                    print(
                        f"Patient directory {pf} not found after extraction, skipping..."
                    )
                    continue

                # Find transcript and columns
                try:
                    tx = next(f for f in os.listdir(pf) if "Transcript" in f)
                    df = pd.read_csv(os.path.join(pf, tx))
                    df.columns = [c.lower() for c in df.columns]
                    start_c = next(c for c in df.columns if "start" in c)
                    end_c = next(c for c in df.columns if "end" in c)
                except (StopIteration, FileNotFoundError) as e:
                    print(f"Error processing transcript for patient {pid}: {e}")
                    continue

                # Read audio and check sample rate
                try:
                    wav0 = next(f for f in os.listdir(pf) if f.lower().endswith(".wav"))
                    y, orig_sr = sf.read(os.path.join(pf, wav0))
                    # Skip 48kHz audio for now
                    if orig_sr == 48000:
                        print(f"Skipping patient {pid}: audio is 48kHz")
                        continue
                    # Resample if needed (but not 48kHz)
                    if orig_sr != DEFAULT_SR_TARGET:
                        y = librosa.resample(
                            y, orig_sr=orig_sr, target_sr=DEFAULT_SR_TARGET
                        )
                    total = len(y)
                except (StopIteration, FileNotFoundError, sf.SoundFileError) as e:
                    print(f"Error processing audio for patient {pid}: {e}")
                    continue

                # First, extract all utterances
                utterances = []
                for i, row in df.iterrows():
                    s = int(row[start_c] * DEFAULT_SR_TARGET)
                    e = int(
                        min(
                            total, (row[end_c] + DEFAULT_BUFFER_SEC) * DEFAULT_SR_TARGET
                        )
                    )
                    if (e - s) > DEFAULT_MAX_UTT_SEC * DEFAULT_SR_TARGET:
                        continue
                    utterances.append(y[s:e])

                # Merge all utterances back into a single audio stream
                if utterances:
                    merged_audio = np.concatenate(utterances)

                    # Create 6-second segments with 50% overlap
                    segments = create_overlapping_segments(
                        merged_audio, DEFAULT_SR_TARGET
                    )

                    # Save each segment as a separate file
                    utterance_count = 0
                    for i, segment in enumerate(segments):
                        utterance_filename = f"{pid}_{i:04d}.wav"
                        utterance_path = os.path.join(default_dir, utterance_filename)
                        sf.write(utterance_path, segment, DEFAULT_SR_TARGET)

                        utterance_data = {
                            "filename": utterance_filename,
                            "Participant_ID": pid,
                            "Gender": patient_info["Gender"],
                            "PHQ_Binary": patient_info["PHQ_Binary"],
                            "PHQ_Score": patient_info["PHQ_Score"],
                            "PCL-C (PTSD)": patient_info["PCL-C (PTSD)"],
                            "PTSD Severity": patient_info["PTSD Severity"],
                        }
                        all_utterance_data.append(utterance_data)
                        utterance_count += 1
                else:
                    utterance_count = 0
                    print(f"No valid utterances found for patient {pid}")

                print(
                    f"Successfully processed {utterance_count} segments for patient {pid}"
                )

            except requests.exceptions.RequestException as e:
                print(
                    f"Error downloading/processing {archive_name} for patient {pid}: {e}"
                )
            except tarfile.TarError as e:
                print(f"Error extracting {local_archive_path}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred for patient {pid}: {e}")
            finally:
                if os.path.exists(local_archive_path):
                    os.remove(local_archive_path)

        # 4) Create new CSV files with utterance-level data
        print("Creating utterance-level CSV files...")

        utterance_df = pd.DataFrame(all_utterance_data)

        if utterance_df.empty:
            print("Warning: No utterance data found!")
            return

        for split_file, temp_csv_data in temp_csvs.items():
            split_name = split_file.replace("_split.csv", ".csv")
            split_patients = set(temp_csv_data["Participant_ID"].tolist())

            split_utterances = utterance_df[
                utterance_df["Participant_ID"].isin(split_patients)
            ]

            if not split_utterances.empty:
                output_csv = os.path.join(p, split_name)
                split_utterances.to_csv(output_csv, index=False)
                print(f"Created {output_csv} with {len(split_utterances)} utterances")
            else:
                print(f"Warning: No utterances found for {split_name}")

        for split in ["train_split.csv", "dev_split.csv", "test_split.csv"]:
            temp_csv = os.path.join(p, f"temp_{split}")
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

        print("All patient data processing completed.")

    @property
    def output_dim(self):
        return 2
