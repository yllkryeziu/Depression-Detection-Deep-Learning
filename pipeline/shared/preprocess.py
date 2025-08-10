from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


def segment_audio(waveform: np.ndarray, sr: int, seg_sec: float, overlap_ratio: float) -> List[np.ndarray]:
    seg_samples = int(seg_sec * sr)
    hop = int(seg_samples * (1 - overlap_ratio))
    segments: List[np.ndarray] = []
    start = 0
    while start + seg_samples <= len(waveform):
        segments.append(waveform[start : start + seg_samples])
        start += hop
    if start < len(waveform):
        remaining = waveform[start:]
        if len(remaining) > seg_samples * 0.5:
            padded = np.pad(remaining, (0, seg_samples - len(remaining)))
            segments.append(padded)
    return segments


def compute_mel(segment: np.ndarray, sr: int, cfg: DictConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        win_length=cfg.dataset.win_length,
        n_mels=cfg.dataset.num_mels,
        fmin=cfg.dataset.f_min,
        fmax=cfg.dataset.f_max,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=cfg.dataset.top_db)
    return mel_db.astype(np.float32)


def _pick_patient_speaker(speaker_series: pd.Series) -> str | None:
    values = [str(v).strip().lower() for v in speaker_series.dropna().unique().tolist()]
    candidates = [v for v in values if any(k in v for k in ["participant", "patient", "subject", "p:", "p -", "p "])]
    if candidates:
        return candidates[0]
    non_ellie = [v for v in values if "ellie" not in v and "interviewer" not in v and v not in ("nan", "")]
    return non_ellie[0] if non_ellie else None


@hydra.main(config_path="../cnn10_baseline/conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    raw_default = Path(cfg.paths.data_root) / "default"
    transcripts_dir = Path(cfg.paths.raw_dir) / "transcripts"
    segments_dir = Path(cfg.paths.segments_dir)
    mels_dir = Path(cfg.paths.mels_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)
    mels_dir.mkdir(parents=True, exist_ok=True)

    csvs = {
        "train.csv": Path(cfg.paths.data_root) / "train.csv",
        "dev.csv": Path(cfg.paths.data_root) / "dev.csv",
        "test.csv": Path(cfg.paths.data_root) / "test.csv",
    }

    for name, csv_path in csvs.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        new_rows = []
        for pid in tqdm(df["Participant_ID"].unique(), desc=f"{name} patients"):
            wav_path = raw_default / f"{pid}_AUDIO.wav"
            tx_path = transcripts_dir / f"{pid}_Transcript.csv"
            if not wav_path.exists() or not tx_path.exists():
                continue
            try:
                y, orig_sr = sf.read(wav_path)
                if orig_sr == 48000:
                    continue
                if orig_sr != cfg.dataset.sample_rate:
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=cfg.dataset.sample_rate)

                tx = pd.read_csv(tx_path)
                tx.columns = [c.lower() for c in tx.columns]
                start_c = next((c for c in tx.columns if "start" in c), None)
                end_c = next((c for c in tx.columns if "end" in c), None)
                speaker_c = next((c for c in tx.columns if "speaker" in c or "talker" in c), None)
                if start_c is None or end_c is None or speaker_c is None:
                    continue

                patient_label = _pick_patient_speaker(tx[speaker_c])
                if patient_label is None:
                    continue

                utterances: List[np.ndarray] = []
                for _, row in tx.iterrows():
                    spk = str(row[speaker_c]).strip().lower()
                    if spk is None or patient_label not in spk:
                        continue
                    s = int(float(row[start_c]) * cfg.dataset.sample_rate)
                    e = int(float(row[end_c]) * cfg.dataset.sample_rate + cfg.dataset.buffer_sec * cfg.dataset.sample_rate)
                    s = max(0, s)
                    e = min(len(y), e)
                    if e <= s:
                        continue
                    if (e - s) > int(cfg.dataset.max_utt_sec * cfg.dataset.sample_rate):
                        continue
                    utterances.append(y[s:e])

                if not utterances:
                    continue

                merged = np.concatenate(utterances)
                segments = segment_audio(merged, cfg.dataset.sample_rate, cfg.dataset.segment_sec, cfg.dataset.overlap_ratio)

                for i, seg in enumerate(segments):
                    seg_fn = f"{pid}_{i:04d}.wav"
                    seg_path = segments_dir / seg_fn
                    sf.write(seg_path, seg, cfg.dataset.sample_rate)
                    mel = compute_mel(seg, cfg.dataset.sample_rate, cfg)
                    mel_fn = seg_fn.replace(".wav", ".npy")
                    mel_path = mels_dir / mel_fn
                    np.save(mel_path, mel)
                    meta = df[df["Participant_ID"] == pid].iloc[0].to_dict()
                    meta.update({
                        "segment_path": str(seg_path),
                        "mel_path": str(mel_path),
                        "segment_index": i,
                    })
                    new_rows.append(meta)
            except Exception:
                continue

        out_df = pd.DataFrame(new_rows)
        out_csv = Path(cfg.paths.data_root) / name
        out_df.to_csv(out_csv, index=False)

    print("Preprocess step finished.")


if __name__ == "__main__":
    main()


