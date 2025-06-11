#!/usr/bin/env python3
"""
Custom preprocessing script for AVEC 2019 style mel spectrograms.
Generates 4-second sliding windows with 1-second hop and stores 
multiple spectrograms per patient in .npy files.

This bypasses autrainer's preprocessing to create proper temporal sequences.
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def load_audio(file_path, target_sr=16000):
    """Load audio file and resample to target sample rate."""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def create_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512, fmin=50, fmax=8000):
    """Create mel spectrogram from audio."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def spectrogram_to_rgb_image(mel_spec_db, height=128, width=160, cmap='magma'):
    """
    Convert mel spectrogram to RGB image following autrainer's SpectToImage transform.
    
    Args:
        mel_spec_db: Mel spectrogram in dB scale
        height: Target height (frequency bins)
        width: Target width (time bins) 
        cmap: Colormap to use
    
    Returns:
        RGB image tensor (3, height, width) as uint8
    """
    # Normalize to [0, 1]
    mel_norm = mel_spec_db.copy()
    mel_norm -= mel_norm.min()
    mel_norm /= mel_norm.max()
    
    # Apply colormap
    cmap_func = get_cmap(cmap)
    colored = np.uint8(cmap_func(mel_norm)[..., :3] * 255)
    
    # Convert to PIL Image and resize
    im = Image.fromarray(colored)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)  # Flip to match autrainer
    im = im.resize((width, height))
    
    # Convert to tensor format (3, height, width)
    rgb_array = np.array(im)
    rgb_tensor = rgb_array.transpose(2, 0, 1).astype(np.uint8)
    
    return rgb_tensor


def create_sliding_windows(audio, sr, window_duration=4.0, hop_duration=1.0):
    """
    Create sliding windows from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        window_duration: Window duration in seconds
        hop_duration: Hop duration in seconds
    
    Returns:
        List of audio windows
    """
    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)
    
    windows = []
    start = 0
    
    while start + window_samples <= len(audio):
        window = audio[start:start + window_samples]
        windows.append(window)
        start += hop_samples
    
    return windows


def process_patient_audio(audio_file, output_file, window_duration=4.0, hop_duration=1.0, 
                         target_sr=16000, n_mels=128, height=128, width=160):
    """
    Process single patient audio file to create sequence of mel spectrograms.
    
    Args:
        audio_file: Path to input audio file
        output_file: Path to output .npy file
        window_duration: Window duration in seconds (default 4.0 for AVEC 2019)
        hop_duration: Hop duration in seconds (default 1.0 for AVEC 2019)
        target_sr: Target sample rate
        n_mels: Number of mel frequency bins
        height: Output image height
        width: Output image width
    
    Returns:
        Number of windows created
    """
    # Load audio
    audio, sr = load_audio(audio_file, target_sr)
    
    # Create sliding windows
    windows = create_sliding_windows(audio, sr, window_duration, hop_duration)
    
    if len(windows) == 0:
        print(f"Warning: No windows created for {audio_file} (audio too short)")
        return 0
    
    # Process each window to create mel spectrograms
    spectrograms = []
    
    for window in windows:
        # Create mel spectrogram
        mel_spec_db = create_mel_spectrogram(window, sr, n_mels=n_mels)
        
        # Convert to RGB image
        rgb_image = spectrogram_to_rgb_image(mel_spec_db, height=height, width=width)
        
        spectrograms.append(rgb_image)
    
    # Stack into array: (num_windows, 3, height, width)
    spectrograms_array = np.stack(spectrograms, axis=0)
    
    # Save to .npy file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, spectrograms_array)
    
    return len(windows)


def main():
    input_dir = Path('data/ExtendedDAIC/default')
    output_dir = Path('data/ExtendedDAIC/mel_spectrograms_avec2019')
    window_duration = 4.0  # 4 seconds for AVEC 2019
    hop_duration = 1.0     # 1 second hop for AVEC 2019
    target_sr = 16000      # 16kHz sample rate
    n_mels = 128           # 128 mel frequency bins
    height = 128           # Image height (frequency dimension)
    width = 160            # Image width (time dimension for 4s window)
    overwrite = False      # Set to True to overwrite existing files
    
    print(f"Processing audio files from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window: {window_duration}s, Hop: {hop_duration}s")
    print(f"Expected time bins for 4s window: {width}")
    
    # Find all patient directories
    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    patient_dirs.sort()
    
    total_windows = 0
    processed_patients = 0
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name
        
        # Find audio file (assuming format: {patient_id}_P.wav)
        audio_file = patient_dir / f"{patient_id}_P.wav"
        
        if not audio_file.exists():
            print(f"Warning: Audio file not found for patient {patient_id}")
            continue
        
        # Output file
        output_file = output_dir / patient_id / f"{patient_id}_P.npy"
        
        # Skip if file exists and not overwriting
        if output_file.exists() and not overwrite:
            print(f"Skipping {patient_id} (file exists)")
            continue
        
        # Process patient
        try:
            num_windows = process_patient_audio(
                audio_file=audio_file,
                output_file=output_file,
                window_duration=window_duration,
                hop_duration=hop_duration,
                target_sr=target_sr,
                n_mels=n_mels,
                height=height,
                width=width
            )
            
            total_windows += num_windows
            processed_patients += 1
            
            # Load audio duration for verification
            audio, sr = librosa.load(audio_file, sr=None)
            duration = len(audio) / sr
            
            print(f"Patient {patient_id}: {num_windows} windows from {duration:.1f}s audio")
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Processed {processed_patients} patients")
    print(f"Generated {total_windows} total windows")
    if processed_patients > 0:
        print(f"Average windows per patient: {total_windows/processed_patients:.1f}")
    
    # Verification: Load a sample file to check format
    if processed_patients > 0:
        sample_file = output_dir / patient_dirs[0].name / f"{patient_dirs[0].name}_P.npy"
        if sample_file.exists():
            sample_data = np.load(sample_file)
            print(f"\nSample output shape: {sample_data.shape}")
            print(f"Expected format: (num_windows, 3, {height}, {width})")
            print(f"Data type: {sample_data.dtype}")


if __name__ == "__main__":
    main() 