# Depression Detection on ExtendedDAIC using CNN+LSTM approach

This pipeline downloads ExtendedDAIC, filters to patient-only speech using transcripts, splits into segments, builds mels, and provides two approaches organized as separate modules:
- `cnn10_baseline/`: Segment-level CNN10 classification (+ optional majority vote to patient). Uses its own Hydra configs under `cnn10_baseline/conf/`.
- `cnn10_lstm/`: Patient-level LSTM over CNN10 segment features. Uses its own Hydra configs under `cnn10_lstm/conf/`.

### Approach
- CNN10 is fine-tuned end-to-end on mel spectrogram segments to predict depression at the segment level.
- The fine-tuned CNN10 is then used as a frozen feature extractor to produce a 512-d feature per segment and write it back to the CSVs (`feature_path`).
- An LSTM consumes per-patient sequences of these segment features (grouped by `Participant_ID`) to predict depression at the patient level.

### Setup
- Create `.env` with `BASE_URL` pointing to dataset host.
- Optionally create `patient_ids.txt` with one `Participant_ID` per line. Otherwise default ranges in `conf/dataset.yaml` are used.
- Ensure requirements are installed.

### Configuration
Each approach has its own configs:
- Baseline: `cnn10_baseline/conf/`
- Patient LSTM: `cnn10_lstm/conf/`

Use hydra overrides as needed, e.g. `train.epochs=80 dataset.segment_sec=8.0`.

### Steps
1) Download
```bash
python -m shared.download paths.data_root=/abs/path/to/data
```
Creates `raw/`, `patients/`, `default/`, and `raw/transcripts/`. Writes `train.csv`, `dev.csv`, `test.csv` with utterance-level rows.

2) Preprocess
```bash
python -m shared.preprocess paths.data_root=/abs/path/to/data
```
Parses transcripts to keep only the patient's utterances, concatenates them, then produces `segments/` wavs and `mels/` npy files. Updates CSVs with `segment_path`, `mel_path`, `segment_index`.

3) Train and extract features
```bash
python -m cnn10_baseline.train model=cnn10 paths.data_root=/abs/path/to/data
python -m shared.feature_extraction paths.data_root=/abs/path/to/data
python -m cnn10_lstm.train model=patient_lstm paths.data_root=/abs/path/to/data
```
This trains CNN10 on segments, extracts segment features using the trained CNN10, and then trains the patient-level LSTM.
Saves best checkpoints to `runs/cnn10_best.pt` and `runs/patient_lstm_best.pt`.

4) Evaluate
```bash
python -m cnn10_baseline.evaluate paths.data_root=/abs/path/to/data eval.ckpt_path=/path/to/cnn10_best.pt
python -m cnn10_baseline.majority_vote paths.data_root=/abs/path/to/data
python -m cnn10_lstm.evaluate model=patient_lstm paths.data_root=/abs/path/to/data eval.ckpt_path=/path/to/patient_lstm_best.pt
```
Prints metrics and logs to W&B if enabled.

### Notes
- Audio is resampled to 16 kHz. 48 kHz inputs are skipped.
- Segments are of fixed length with configurable overlap. Short tail segments are padded.
- Mel spectrogram parameters are configurable in `dataset.yaml`.
- Transcript parsing auto-detects `speaker`/`talker` column and selects the participant/patient rows (heuristics for names like "participant", "patient", "subject", or not "ellie").


