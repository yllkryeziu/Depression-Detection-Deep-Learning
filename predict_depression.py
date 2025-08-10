#!/usr/bin/env python3
"""
Depression Prediction Script using CNN10 Model

This script loads a trained CNN10 model and predicts depression classification
on test mel-spectrograms with patient-level majority voting.

Usage:
    python predict_depression.py
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block for CNN10."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
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


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class AbstractModel(nn.Module):
    """Abstract base model class."""
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim


class Cnn10New(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        feature_extractor_freeze: bool = False,
        dropout_rate: float = 0.5,
        hidden_dim: int = 256,
        in_channels: int = 1,
        transfer: Optional[str] = None,
    ) -> None:
        """CNN10-New model for binary depression classification using mel-spectrograms."""
        super().__init__(output_dim)
        self.feature_extractor_freeze = feature_extractor_freeze
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.transfer = transfer
        
        # CNN10 feature extractor layers
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        
        # Feature processing layer (from original CNN10)
        self.fc1 = torch.nn.Linear(512, 512, bias=True)
        
        # Custom classification head for depression detection
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(512, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(self.hidden_dim // 2, output_dim)
        )

        self.init_weight()
        
        # Freeze feature extractor if requested
        if self.feature_extractor_freeze:
            self._freeze_feature_extractor()

    def init_weight(self) -> None:
        """Initialize model weights."""
        init_bn(self.bn0)
        init_layer(self.fc1)
        
        # Initialize classification head
        for layer in self.classification_head:
            if isinstance(layer, torch.nn.Linear):
                init_layer(layer)
    
    def _freeze_feature_extractor(self) -> None:
        """Freeze the CNN10 feature extractor weights."""
        for param in self.bn0.parameters():
            param.requires_grad = False
        for param in self.conv_block1.parameters():
            param.requires_grad = False
        for param in self.conv_block2.parameters():
            param.requires_grad = False
        for param in self.conv_block3.parameters():
            param.requires_grad = False
        for param in self.conv_block4.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False

    def extract_cnn10_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using CNN10 architecture."""
        # CNN10 feature extraction pipeline
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        # Clipwise aggregation (global pooling)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input features."""
        return self.extract_cnn10_features(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Extract features using CNN10
        x = self.embeddings(features)
        
        # Apply classification head for depression detection
        x = self.classification_head(x)
        
        return x


class SpectrogramDataset(Dataset):
    """Dataset for loading spectrogram files."""
    
    def __init__(self, filenames: List[str], spectrogram_dir: str):
        self.filenames = filenames
        self.spectrogram_dir = spectrogram_dir
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # Convert .wav filename to .npy filename
        npy_filename = filename.replace('.wav', '.npy')
        filepath = os.path.join(self.spectrogram_dir, npy_filename)
        
        # Load spectrogram as numpy array
        try:
            spectrogram = np.load(filepath)
            
            # Convert to tensor and add channel dimension if needed
            if len(spectrogram.shape) == 2:
                spectrogram = spectrogram[np.newaxis, :, :]  # Add channel dimension
            
            return torch.FloatTensor(spectrogram), filename
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            # Return a dummy tensor in case of error
            return torch.zeros((1, 128, 401)), filename


def load_model(model_path: str, device: str) -> Cnn10New:
    """Load the trained CNN10 model."""
    logger.info(f"Loading model from {model_path}")
    
    # Initialize model with same parameters as training
    model = Cnn10New(
        output_dim=2,  # Binary classification
        feature_extractor_freeze=False,
        dropout_rate=0.5,
        hidden_dim=256,
        in_channels=1
    )
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def predict_patient_snippets(
    patient_id: int,
    patient_filenames: List[str],
    model: Cnn10New,
    spectrogram_dir: str,
    device: str,
    batch_size: int = 32
) -> List[Dict]:
    """Predict depression for all snippets of a patient."""
    
    # Create dataset and dataloader for this patient
    dataset = SpectrogramDataset(patient_filenames, spectrogram_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    predictions = []
    
    with torch.no_grad():
        for batch_spectrograms, batch_filenames in dataloader:
            batch_spectrograms = batch_spectrograms.to(device)
            
            # Get model outputs (logits)
            outputs = model(batch_spectrograms)
            
            # Convert to probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predictions (class with highest probability)
            predicted_classes = torch.argmax(outputs, dim=1)
            
            # Store results for each snippet
            for i, filename in enumerate(batch_filenames):
                predictions.append({
                    'patient_id': patient_id,
                    'filename': filename,
                    'predicted_class': predicted_classes[i].item(),
                    'prob_not_depressed': probabilities[i][0].item(),
                    'prob_depressed': probabilities[i][1].item(),
                    'confidence': torch.max(probabilities[i]).item()
                })
    
    return predictions


def majority_voting(snippet_predictions: List[Dict]) -> Dict:
    """Perform majority voting for patient-level prediction."""
    if not snippet_predictions:
        return {
            'patient_id': None,
            'num_snippets': 0,
            'predicted_class': 0,
            'confidence': 0.0,
            'vote_distribution': {'not_depressed': 0, 'depressed': 0}
        }
    
    patient_id = snippet_predictions[0]['patient_id']
    
    # Count votes
    votes = [pred['predicted_class'] for pred in snippet_predictions]
    vote_counts = Counter(votes)
    
    # Get majority prediction
    majority_class = vote_counts.most_common(1)[0][0]
    
    # Calculate average confidence for the majority class
    majority_confidences = [
        pred['confidence'] for pred in snippet_predictions 
        if pred['predicted_class'] == majority_class
    ]
    avg_confidence = np.mean(majority_confidences) if majority_confidences else 0.0
    
    # Calculate average probabilities
    avg_prob_not_depressed = np.mean([pred['prob_not_depressed'] for pred in snippet_predictions])
    avg_prob_depressed = np.mean([pred['prob_depressed'] for pred in snippet_predictions])
    
    return {
        'patient_id': patient_id,
        'num_snippets': len(snippet_predictions),
        'predicted_class': majority_class,
        'avg_prob_not_depressed': avg_prob_not_depressed,
        'avg_prob_depressed': avg_prob_depressed,
        'confidence': avg_confidence,
        'vote_distribution': {
            'not_depressed': vote_counts.get(0, 0),
            'depressed': vote_counts.get(1, 0)
        }
    }


def main():
    # Configuration
    MODEL_PATH = "model.pt"
    TEST_CSV_PATH = "data/ExtendedDAIC-16k-fixed/test.csv"
    SPECTROGRAM_DIR = "data/ExtendedDAIC-16k-fixed/log_mel_16k"
    OUTPUT_FILE = "depression_predictions.csv"
    BATCH_SIZE = 32
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(TEST_CSV_PATH):
        logger.error(f"Test CSV file not found: {TEST_CSV_PATH}")
        return
    
    if not os.path.exists(SPECTROGRAM_DIR):
        logger.error(f"Spectrogram directory not found: {SPECTROGRAM_DIR}")
        return
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Group by patient
    patient_groups = test_df.groupby('Participant_ID')
    logger.info(f"Found {len(patient_groups)} unique patients")
    
    # Predict for each patient
    all_snippet_predictions = []
    patient_predictions = []
    
    for patient_id, patient_data in patient_groups:
        logger.info(f"Processing patient {patient_id} ({len(patient_data)} snippets)")
        
        # Get filenames for this patient
        patient_filenames = patient_data['filename'].tolist()
        
        # Predict for all snippets of this patient
        snippet_preds = predict_patient_snippets(
            patient_id, patient_filenames, model, SPECTROGRAM_DIR, device, BATCH_SIZE
        )
        
        all_snippet_predictions.extend(snippet_preds)
        
        # Perform majority voting
        patient_pred = majority_voting(snippet_preds)
        
        # Add ground truth information
        ground_truth = patient_data.iloc[0]
        patient_pred.update({
            'ground_truth_class': ground_truth['PHQ_Binary'],
            'ground_truth_score': ground_truth['PHQ_Score'],
            'gender': ground_truth['Gender']
        })
        
        patient_predictions.append(patient_pred)
        
        logger.info(f"Patient {patient_id}: Predicted={patient_pred['predicted_class']}, "
                   f"Ground Truth={patient_pred['ground_truth_class']}, "
                   f"Confidence={patient_pred['confidence']:.3f}")
    
    # Save snippet-level predictions
    snippet_df = pd.DataFrame(all_snippet_predictions)
    snippet_output_file = "snippet_predictions.csv"
    snippet_df.to_csv(snippet_output_file, index=False)
    logger.info(f"Saved snippet-level predictions to {snippet_output_file}")
    
    # Save patient-level predictions
    patient_df = pd.DataFrame(patient_predictions)
    patient_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved patient-level predictions to {OUTPUT_FILE}")
    
    # Calculate accuracy
    correct_predictions = sum(1 for pred in patient_predictions 
                            if pred['predicted_class'] == pred['ground_truth_class'])
    total_patients = len(patient_predictions)
    accuracy = correct_predictions / total_patients if total_patients > 0 else 0
    
    logger.info(f"\nResults Summary:")
    logger.info(f"Total patients: {total_patients}")
    logger.info(f"Correct predictions: {correct_predictions}")
    logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Class-wise statistics
    depressed_patients = [p for p in patient_predictions if p['ground_truth_class'] == 1]
    not_depressed_patients = [p for p in patient_predictions if p['ground_truth_class'] == 0]
    
    if depressed_patients:
        depressed_correct = sum(1 for p in depressed_patients if p['predicted_class'] == 1)
        depressed_accuracy = depressed_correct / len(depressed_patients)
        logger.info(f"Depressed patients accuracy: {depressed_accuracy:.3f} ({len(depressed_patients)} patients)")
    
    if not_depressed_patients:
        not_depressed_correct = sum(1 for p in not_depressed_patients if p['predicted_class'] == 0)
        not_depressed_accuracy = not_depressed_correct / len(not_depressed_patients)
        logger.info(f"Not depressed patients accuracy: {not_depressed_accuracy:.3f} ({len(not_depressed_patients)} patients)")
    
    logger.info(f"\nPrediction files saved:")
    logger.info(f"- Patient-level: {OUTPUT_FILE}")
    logger.info(f"- Snippet-level: {snippet_output_file}")


if __name__ == "__main__":
    main() 