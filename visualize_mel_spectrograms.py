#!/usr/bin/env python3
"""
Mel Spectrogram Visualization Script for Depression Detection

This script visualizes the mel spectrograms used in your CNN+GRU depression detection model.
It shows how audio is converted to visual features that CNNs can process.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import random
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MelSpectrogramVisualizer:
    """
    Visualize mel spectrograms from the ExtendedDAIC depression detection dataset.
    """
    
    def __init__(self, data_path: str = "data/ExtendedDAIC"):
        self.data_path = Path(data_path)
        self.mel_path = self.data_path / "mel_spectrograms_avec2019"
        self.train_df = pd.read_csv(self.data_path / "train.csv")
        self.dev_df = pd.read_csv(self.data_path / "dev.csv")
        self.test_df = pd.read_csv(self.data_path / "test.csv")
        
        # Combine all dataframes for comprehensive analysis
        self.all_df = pd.concat([self.train_df, self.dev_df, self.test_df], ignore_index=True)
        
    def load_mel_spectrogram(self, participant_id: int) -> np.ndarray:
        """Load mel spectrogram for a specific participant."""
        mel_file = self.mel_path / str(participant_id) / f"{participant_id}_P.npy"
        if mel_file.exists():
            return np.load(mel_file)
        else:
            raise FileNotFoundError(f"Mel spectrogram not found for participant {participant_id}")
    
    def analyze_data_structure(self) -> None:
        """Analyze and print the structure of mel spectrogram data."""
        print("=" * 60)
        print("MEL SPECTROGRAM DATA STRUCTURE ANALYSIS")
        print("=" * 60)
        
        # Sample a few participants
        sample_ids = self.all_df['Participant_ID'].sample(5).tolist()
        
        for pid in sample_ids:
            try:
                mel_data = self.load_mel_spectrogram(pid)
                label = self.all_df[self.all_df['Participant_ID'] == pid]['PHQ_Binary'].iloc[0]
                
                print(f"\nParticipant {pid} (Depression: {'Yes' if label == 1 else 'No'}):")
                print(f"  Shape: {mel_data.shape}")
                print(f"  Data type: {mel_data.dtype}")
                print(f"  Value range: [{mel_data.min():.3f}, {mel_data.max():.3f}]")
                print(f"  Memory size: {mel_data.nbytes / 1024:.1f} KB")
                
                if len(mel_data.shape) == 4:  # [seq_len, channels, height, width]
                    seq_len, channels, height, width = mel_data.shape
                    print(f"  Sequence length: {seq_len} frames")
                    print(f"  Channels: {channels} (RGB)")
                    print(f"  Height: {height} mel bands")
                    print(f"  Width: {width} time bins per frame")
                    print(f"  Total duration: ~{seq_len} seconds (1-second hop)")
                    
            except FileNotFoundError:
                print(f"  Participant {pid}: File not found")
        
        print("\n" + "=" * 60)
    
    def visualize_single_frame(self, participant_id: int, frame_idx: int = 0) -> None:
        """Visualize a single mel spectrogram frame (4-second window)."""
        mel_data = self.load_mel_spectrogram(participant_id)
        label = self.all_df[self.all_df['Participant_ID'] == participant_id]['PHQ_Binary'].iloc[0]
        label_text = "Depressed" if label == 1 else "Not Depressed"
        
        if len(mel_data.shape) == 4:
            frame = mel_data[frame_idx]  # [channels, height, width]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Participant {participant_id} - Frame {frame_idx} - {label_text}', 
                        fontsize=16, fontweight='bold')
            
            # Individual color channels
            channel_names = ['Red', 'Green', 'Blue']
            colors = ['Reds', 'Greens', 'Blues']
            
            for i in range(3):
                row, col = i // 2, i % 2
                if i < 2:
                    im = axes[row, col].imshow(frame[i], aspect='auto', origin='lower', 
                                             cmap=colors[i], interpolation='nearest')
                    axes[row, col].set_title(f'{channel_names[i]} Channel')
                    axes[row, col].set_xlabel('Time Bins (4-second window)')
                    axes[row, col].set_ylabel('Mel Frequency Bands')
                    plt.colorbar(im, ax=axes[row, col])
            
            # Combined RGB visualization
            # Transpose from [C, H, W] to [H, W, C] for matplotlib
            rgb_frame = np.transpose(frame, (1, 2, 0))
            # Normalize to [0, 1] for RGB display
            rgb_frame = (rgb_frame - rgb_frame.min()) / (rgb_frame.max() - rgb_frame.min())
            
            axes[1, 1].imshow(rgb_frame, aspect='auto', origin='lower', interpolation='nearest')
            axes[1, 1].set_title('Combined RGB (Magma Colormap)')
            axes[1, 1].set_xlabel('Time Bins (4-second window)')
            axes[1, 1].set_ylabel('Mel Frequency Bands')
            
            plt.tight_layout()
            plt.show()
            
            # Print frame statistics
            print(f"\nFrame {frame_idx} Statistics:")
            print(f"  Shape: {frame.shape}")
            print(f"  Time bins: {frame.shape[2]} (≈4 seconds)")
            print(f"  Mel bands: {frame.shape[1]} (50-8000 Hz)")
            print(f"  Value range: [{frame.min():.3f}, {frame.max():.3f}]")
    
    def visualize_temporal_sequence(self, participant_id: int, max_frames: int = 10) -> None:
        """Visualize the temporal sequence of mel spectrograms."""
        mel_data = self.load_mel_spectrogram(participant_id)
        label = self.all_df[self.all_df['Participant_ID'] == participant_id]['PHQ_Binary'].iloc[0]
        label_text = "Depressed" if label == 1 else "Not Depressed"
        
        if len(mel_data.shape) == 4:
            seq_len = min(mel_data.shape[0], max_frames)
            
            fig, axes = plt.subplots(2, seq_len // 2, figsize=(20, 8))
            if seq_len == 1:
                axes = [axes]
            elif seq_len <= 2:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Temporal Sequence - Participant {participant_id} - {label_text}', 
                        fontsize=16, fontweight='bold')
            
            for i in range(seq_len):
                row, col = i // (seq_len // 2), i % (seq_len // 2)
                if seq_len <= 2:
                    row = 0
                
                # Use RGB combined view
                frame = mel_data[i]
                rgb_frame = np.transpose(frame, (1, 2, 0))
                rgb_frame = (rgb_frame - rgb_frame.min()) / (rgb_frame.max() - rgb_frame.min())
                
                axes[row, col].imshow(rgb_frame, aspect='auto', origin='lower', interpolation='nearest')
                axes[row, col].set_title(f'Frame {i} (t={i}s)')
                axes[row, col].set_xlabel('Time Bins')
                axes[row, col].set_ylabel('Mel Bands')
                axes[row, col].tick_params(labelsize=8)
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nTemporal Sequence Analysis:")
            print(f"  Total frames: {mel_data.shape[0]}")
            print(f"  Displayed frames: {seq_len}")
            print(f"  Each frame: 4-second audio window")
            print(f"  Frame overlap: 3 seconds (1-second hop)")
            print(f"  Total audio duration: ≈{mel_data.shape[0]} seconds")
    
    def compare_depression_vs_control(self) -> None:
        """Compare mel spectrograms between depressed and non-depressed participants."""
        # Get samples from each class
        depressed = self.all_df[self.all_df['PHQ_Binary'] == 1]['Participant_ID'].sample(2).tolist()
        control = self.all_df[self.all_df['PHQ_Binary'] == 0]['Participant_ID'].sample(2).tolist()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Depression vs Control Comparison (First Frame)', fontsize=16, fontweight='bold')
        
        participants = [*depressed, *control]
        labels = ['Depressed', 'Depressed', 'Control', 'Control']
        
        for i, (pid, label) in enumerate(zip(participants, labels)):
            try:
                mel_data = self.load_mel_spectrogram(pid)
                frame = mel_data[0]  # First frame
                
                # RGB combined view
                rgb_frame = np.transpose(frame, (1, 2, 0))
                rgb_frame = (rgb_frame - rgb_frame.min()) / (rgb_frame.max() - rgb_frame.min())
                
                row, col = i // 4, i % 4
                axes[row, col].imshow(rgb_frame, aspect='auto', origin='lower', interpolation='nearest')
                axes[row, col].set_title(f'Participant {pid}\n{label}')
                axes[row, col].set_xlabel('Time Bins')
                axes[row, col].set_ylabel('Mel Bands')
                
            except FileNotFoundError:
                axes[row, col].text(0.5, 0.5, f'Participant {pid}\nNot Found', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'Participant {pid}\n{label}')
        
        plt.tight_layout()
        plt.show()
        
        print("\nComparison Notes:")
        print("- Look for patterns in energy distribution across frequency bands")
        print("- Depression may show different temporal dynamics")
        print("- Lower frequencies often contain prosodic information")
        print("- Higher frequencies contain phonetic details")
    
    def analyze_frequency_distribution(self, participant_id: int) -> None:
        """Analyze energy distribution across frequency bands."""
        mel_data = self.load_mel_spectrogram(participant_id)
        label = self.all_df[self.all_df['Participant_ID'] == participant_id]['PHQ_Binary'].iloc[0]
        label_text = "Depressed" if label == 1 else "Not Depressed"
        
        if len(mel_data.shape) == 4:
            # Average across time and channels to get frequency profile
            # Shape: [seq_len, channels, height, width] -> [height]
            freq_profile = np.mean(mel_data, axis=(0, 2, 3))  # Average over time, channels, and time bins
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Frequency Analysis - Participant {participant_id} - {label_text}', 
                        fontsize=14, fontweight='bold')
            
            # Frequency profile
            mel_bands = np.arange(len(freq_profile))
            ax1.plot(mel_bands, freq_profile, linewidth=2, color='darkblue')
            ax1.fill_between(mel_bands, freq_profile, alpha=0.3, color='lightblue')
            ax1.set_xlabel('Mel Frequency Band')
            ax1.set_ylabel('Average Energy')
            ax1.set_title('Energy Distribution Across Frequency Bands')
            ax1.grid(True, alpha=0.3)
            
            # Temporal energy variation
            temporal_energy = np.mean(mel_data, axis=(1, 2, 3))  # Average over channels, height, width
            time_points = np.arange(len(temporal_energy))
            ax2.plot(time_points, temporal_energy, linewidth=2, color='darkgreen', marker='o')
            ax2.set_xlabel('Time Frame (seconds)')
            ax2.set_ylabel('Average Energy')
            ax2.set_title('Energy Variation Over Time')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nFrequency Analysis:")
            print(f"  Peak energy at mel band: {np.argmax(freq_profile)}")
            print(f"  Energy range: [{freq_profile.min():.3f}, {freq_profile.max():.3f}]")
            print(f"  Temporal energy std: {np.std(temporal_energy):.3f}")
    
    def create_summary_visualization(self) -> None:
        """Create a comprehensive summary visualization."""
        print("\nCreating comprehensive mel spectrogram summary...")
        
        # Sample participants from different classes
        try:
            depressed_sample = self.all_df[self.all_df['PHQ_Binary'] == 1]['Participant_ID'].iloc[0]
            control_sample = self.all_df[self.all_df['PHQ_Binary'] == 0]['Participant_ID'].iloc[0]
            
            print(f"Analyzing participants: {depressed_sample} (depressed), {control_sample} (control)")
            
            # Analyze data structure
            self.analyze_data_structure()
            
            # Visualize single frames
            print(f"\n--- Single Frame Analysis (Participant {depressed_sample}) ---")
            self.visualize_single_frame(depressed_sample, frame_idx=0)
            
            # Visualize temporal sequence
            print(f"\n--- Temporal Sequence Analysis (Participant {control_sample}) ---")
            self.visualize_temporal_sequence(control_sample, max_frames=8)
            
            # Compare classes
            print(f"\n--- Class Comparison ---")
            self.compare_depression_vs_control()
            
            # Frequency analysis
            print(f"\n--- Frequency Analysis (Participant {depressed_sample}) ---")
            self.analyze_frequency_distribution(depressed_sample)
            
        except Exception as e:
            print(f"Error in summary visualization: {e}")
    
    def explain_mel_spectrograms(self) -> None:
        """Print educational explanation of mel spectrograms."""
        print("\n" + "=" * 80)
        print("UNDERSTANDING MEL SPECTROGRAMS FOR DEPRESSION DETECTION")
        print("=" * 80)
        
        print("""
🎵 WHAT ARE MEL SPECTROGRAMS?
• Visual representation of audio frequency content over time
• Y-axis: Frequency bands (mel scale - perceptually motivated)
• X-axis: Time progression
• Color intensity: Energy/amplitude at that frequency and time

🧠 MEL SCALE ADVANTAGES:
• Mimics human auditory perception
• More resolution at lower frequencies (where speech fundamentals are)
• Better for capturing prosodic features relevant to depression

🖼️  IMAGE CONVERSION (AVEC 2019 APPROACH):
• Convert mel spectrograms to RGB images using magma colormap
• This allows using pretrained ImageNet CNNs!
• Each color channel captures different aspects of the spectral pattern

📊 YOUR DATA STRUCTURE:
• Shape: [sequence_length, 3_channels, 128_mel_bands, 256_time_bins]
• Each frame: 4-second audio window
• Frame overlap: 3 seconds (1-second hop)
• 128 mel bands: 50-8000 Hz range
• 256 time bins: High temporal resolution within each frame

🏥 DEPRESSION MARKERS TO LOOK FOR:
• Prosodic patterns: F0 contours, rhythm, timing
• Voice quality: Breathiness, roughness (higher frequencies)
• Energy distribution: Flat affect may show different energy patterns
• Temporal dynamics: Pauses, speech rate variations

🤖 CNN+GRU PROCESSING:
1. CNN extracts visual features from each mel spectrogram frame
2. GRU models temporal dependencies across frames
3. Final classifier uses last GRU output for depression prediction

💡 KEY INSIGHT:
Treating audio as images leverages decades of computer vision progress
for audio analysis - a brilliant cross-domain transfer!
        """)


def main():
    """Main function to run mel spectrogram visualization."""
    print("🎵 Mel Spectrogram Visualization for Depression Detection 🧠")
    print("=" * 60)
    
    # Initialize visualizer
    try:
        visualizer = MelSpectrogramVisualizer()
        
        # Educational explanation
        visualizer.explain_mel_spectrograms()
        
        # Create comprehensive visualization
        visualizer.create_summary_visualization()
        
        print("\n" + "=" * 60)
        print("✅ Visualization complete! You now understand your mel spectrograms.")
        print("💬 Ready to discuss with your supervisor!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure mel spectrogram data exists in data/ExtendedDAIC/mel_spectrograms_avec2019/")
        print("2. Check that CSV files exist in data/ExtendedDAIC/")
        print("3. Install required packages: matplotlib, seaborn, numpy, pandas")


if __name__ == "__main__":
    main() 