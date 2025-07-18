#!/usr/bin/env python3
"""
CNN Feature Extraction Script for Depression Detection

This script loads a pre-trained CNN10 model, processes spectrogram images for each patient,
extracts feature vectors, and saves them as sequences for CNN-LSTM training.

Usage:
    python extract_cnn_features.py --data_path data/ExtendedDAIC-16k --output_path data/ExtendedDAIC-16k/ --model_path cnn10model.pt

Features:
- Loads pre-trained CNN10 model and removes final classification layer
- Processes spectrograms in patient-specific sequences
- Saves feature sequences as PyTorch tensors (.pt files)
- Creates corrected CSV files with patient-level labels
- GPU acceleration support
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import CNN10 components
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpectrogramDataset(Dataset):
    """Dataset for loading spectrogram files for a specific patient."""
    
    def __init__(self, spectrogram_files: List[str], spectrogram_dir: str):
        self.spectrogram_files = spectrogram_files
        self.spectrogram_dir = spectrogram_dir
    
    def __len__(self):
        return len(self.spectrogram_files)
    
    def __getitem__(self, idx):
        filename = self.spectrogram_files[idx]
        filepath = os.path.join(self.spectrogram_dir, filename)
        
        # Load spectrogram as numpy array
        spectrogram = np.load(filepath)
        
        # Convert to tensor and add channel dimension if needed
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram[np.newaxis, :, :]  # Add channel dimension
        
        return torch.FloatTensor(spectrogram), filename


class ConvBlock(nn.Module):
    """Convolutional block for CNN10."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            x = F.max_pool2d(x, kernel_size=pool_size)
        
        return x


class CNN10FeatureExtractor(nn.Module):
    """CNN10 model with final classification layer removed for feature extraction."""
    
    def __init__(self, pretrained_path: Optional[str] = None, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # CNN10 architecture
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        
        # Initialize weights
        self._init_weights()
        
        # Load pre-trained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pre-trained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.load_state_dict(checkpoint, strict=False)
        else:
            logger.warning("No pre-trained weights provided or file not found. Using random initialization.")
        
        # Set to evaluation mode
        self.eval()
        
        # Move to device
        self.to(device)
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Extract features using CNN10 architecture."""
        with torch.no_grad():
            # CNN10 feature extraction
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

            x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=False)
            x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=False)
            x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=False)
            x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=False)
            x = torch.mean(x, dim=3)

            # Clipwise path (global pooling)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2

            # Final feature layer
            x = F.relu_(self.fc1(x))
            return x


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_columns = ['filename', 'Participant_ID', 'Gender', 'PHQ_Binary', 'PHQ_Score']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    return df


def get_patient_spectrograms(patient_id: int, df: pd.DataFrame, spectrogram_dir: str) -> List[str]:
    """Get list of spectrogram files for a specific patient."""
    patient_files = df[df['Participant_ID'] == patient_id]['filename'].tolist()
    
    # Convert .wav filenames to .npy filenames
    spectrogram_files = []
    for wav_file in patient_files:
        npy_file = wav_file.replace('.wav', '.npy')
        npy_path = os.path.join(spectrogram_dir, npy_file)
        if os.path.exists(npy_path):
            spectrogram_files.append(npy_file)
    
    # Sort files to ensure consistent ordering
    spectrogram_files.sort()
    return spectrogram_files


def extract_patient_features(
    patient_id: int,
    spectrogram_files: List[str],
    feature_extractor: CNN10FeatureExtractor,
    spectrogram_dir: str,
    batch_size: int = 32
) -> torch.Tensor:
    """Extract feature vectors for all spectrograms of a patient."""
    
    if not spectrogram_files:
        logger.warning(f"No spectrogram files found for patient {patient_id}")
        return torch.empty(0, 512)  # CNN10 outputs 512-dim features
    
    # Create dataset and dataloader
    dataset = SpectrogramDataset(spectrogram_files, spectrogram_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    feature_vectors = []
    
    for batch_spectrograms, batch_filenames in dataloader:
        batch_spectrograms = batch_spectrograms.to(feature_extractor.device)
        
        # Extract features
        batch_features = feature_extractor(batch_spectrograms)
        feature_vectors.append(batch_features.cpu())
    
    # Concatenate all feature vectors for this patient
    if feature_vectors:
        patient_features = torch.cat(feature_vectors, dim=0)
    else:
        patient_features = torch.empty(0, 512)
    
    return patient_features


def create_patient_level_csv(df: pd.DataFrame, output_path: str, split_name: str):
    """Create patient-level CSV with corrected labels."""
    
    # Group by patient and get first occurrence (all rows for same patient should have same labels)
    patient_df = df.groupby('Participant_ID').agg({
        'Gender': 'first',
        'PHQ_Binary': 'first', 
        'PHQ_Score': 'first',
        'PCL-C (PTSD)': 'first',
        'PTSD Severity': 'first',
        'filename': 'count'  # Count number of segments per patient
    }).reset_index()
    
    # Rename filename count to segment_count
    patient_df.rename(columns={'filename': 'segment_count'}, inplace=True)
    
    # Add feature file column
    patient_df['feature_file'] = patient_df['Participant_ID'].apply(lambda x: f"patient_{x}.pt")
    
    # Save patient-level CSV
    output_csv = os.path.join(output_path, f"{split_name}_patients.csv")
    patient_df.to_csv(output_csv, index=False)
    logger.info(f"Created patient-level CSV: {output_csv} with {len(patient_df)} patients")
    
    return patient_df


def main():
    parser = argparse.ArgumentParser(description="Extract CNN features for CNN-LSTM training")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to dataset directory containing CSV files and log_mel_16k folder")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output directory for feature sequences and corrected CSVs")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pre-trained CNN10 model (.pt or .pth file)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for computation")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")
    
    spectrogram_dir = os.path.join(args.data_path, "log_mel_16k")
    if not os.path.exists(spectrogram_dir):
        raise FileNotFoundError(f"Spectrogram directory not found: {spectrogram_dir}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    features_dir = os.path.join(args.output_path, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Spectrogram directory: {spectrogram_dir}")
    
    # Initialize feature extractor
    logger.info("Initializing CNN10 feature extractor...")
    feature_extractor = CNN10FeatureExtractor(
        pretrained_path=args.model_path,
        device=args.device
    )
    
    # Process each split (train, dev, test)
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        csv_path = os.path.join(args.data_path, f"{split}.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}. Skipping {split} split.")
            continue
        
        logger.info(f"\nProcessing {split} split...")
        
        # Load CSV data
        df = load_csv_data(csv_path)
        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        
        # Get unique patients
        unique_patients = df['Participant_ID'].unique()
        logger.info(f"Found {len(unique_patients)} unique patients in {split} split")
        
        # Extract features for each patient
        processed_patients = []
        failed_patients = []
        
        for patient_id in tqdm(unique_patients, desc=f"Extracting features for {split}"):
            try:
                # Get spectrogram files for this patient
                spectrogram_files = get_patient_spectrograms(patient_id, df, spectrogram_dir)
                
                if not spectrogram_files:
                    logger.warning(f"No spectrograms found for patient {patient_id}")
                    failed_patients.append(patient_id)
                    continue
                
                # Extract features
                patient_features = extract_patient_features(
                    patient_id, spectrogram_files, feature_extractor, 
                    spectrogram_dir, args.batch_size
                )
                
                if patient_features.size(0) == 0:
                    logger.warning(f"No features extracted for patient {patient_id}")
                    failed_patients.append(patient_id)
                    continue
                
                # Save features
                feature_file = os.path.join(features_dir, f"patient_{patient_id}.pt")
                torch.save(patient_features, feature_file)
                
                processed_patients.append(patient_id)
                logger.debug(f"Saved features for patient {patient_id}: shape {patient_features.shape}")
                
            except Exception as e:
                logger.error(f"Error processing patient {patient_id}: {str(e)}")
                failed_patients.append(patient_id)
        
        logger.info(f"Successfully processed {len(processed_patients)} patients for {split} split")
        if failed_patients:
            logger.warning(f"Failed to process {len(failed_patients)} patients: {failed_patients}")
        
        # Create patient-level CSV (only for successfully processed patients)
        if processed_patients:
            # Filter dataframe to only include successfully processed patients
            processed_df = df[df['Participant_ID'].isin(processed_patients)]
            create_patient_level_csv(processed_df, args.output_path, split)
    
    logger.info("\nFeature extraction completed!")
    logger.info(f"Features saved in: {features_dir}")
    logger.info(f"Patient-level CSVs saved in: {args.output_path}")
    
    # Print summary statistics
    feature_files = list(Path(features_dir).glob("patient_*.pt"))
    logger.info(f"Total feature files created: {len(feature_files)}")
    
    if feature_files:
        # Sample a few files to show feature dimensions
        sample_features = torch.load(feature_files[0])
        logger.info(f"Feature dimensions: {sample_features.shape} (sequence_length, feature_dim)")
        logger.info(f"Feature data type: {sample_features.dtype}")


if __name__ == "__main__":
    main() 