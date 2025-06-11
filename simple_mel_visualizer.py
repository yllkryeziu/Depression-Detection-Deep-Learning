#!/usr/bin/env python3
"""
Simple Mel Spectrogram Visualizer for Depression Detection
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_and_visualize_mel_spectrogram(participant_id, data_path="data/ExtendedDAIC"):
    """Load and visualize mel spectrogram for a specific participant."""
    
    # Load the data
    mel_file = Path(data_path) / "mel_spectrograms_avec2019" / str(participant_id) / f"{participant_id}_P.npy"
    
    if not mel_file.exists():
        print(f"❌ Mel spectrogram file not found for participant {participant_id}")
        return None
    
    mel_data = np.load(mel_file)
    
    # Load label information
    train_df = pd.read_csv(Path(data_path) / "train.csv")
    dev_df = pd.read_csv(Path(data_path) / "dev.csv") 
    test_df = pd.read_csv(Path(data_path) / "test.csv")
    all_df = pd.concat([train_df, dev_df, test_df])
    
    participant_data = all_df[all_df['Participant_ID'] == participant_id]
    if len(participant_data) > 0:
        label = participant_data['PHQ_Binary'].iloc[0]
        label_text = "Depressed" if label == 1 else "Not Depressed"
    else:
        label_text = "Unknown"
    
    print(f"\n🎵 PARTICIPANT {participant_id} - {label_text}")
    print("=" * 50)
    print(f"Data Shape: {mel_data.shape}")
    print(f"Data Type: {mel_data.dtype}")
    print(f"Value Range: [{mel_data.min():.3f}, {mel_data.max():.3f}]")
    
    if len(mel_data.shape) == 4:
        seq_len, channels, height, width = mel_data.shape
        print(f"📊 Structure:")
        print(f"  • Sequence Length: {seq_len} frames (≈{seq_len} seconds)")
        print(f"  • Channels: {channels} (RGB from magma colormap)")
        print(f"  • Height: {height} mel frequency bands (50-8000 Hz)")
        print(f"  • Width: {width} time bins per 4-second frame")
        
        # Visualize the first frame
        frame = mel_data[0]  # First frame [channels, height, width]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Participant {participant_id} - {label_text}\nMel Spectrogram Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Individual RGB channels
        channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
        cmaps = ['Reds', 'Greens', 'Blues']
        
        for i in range(3):
            if i < 2:  # First two channels in top row
                ax = axes[0, i] if i == 0 else axes[0, 1]
                im = ax.imshow(frame[i], aspect='auto', origin='lower', 
                              cmap=cmaps[i], interpolation='nearest')
                ax.set_title(channel_names[i])
                ax.set_xlabel('Time Bins (4-second window)')
                ax.set_ylabel('Mel Frequency Bands')
                plt.colorbar(im, ax=ax)
        
        # Combined RGB visualization
        rgb_frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
        rgb_frame = (rgb_frame - rgb_frame.min()) / (rgb_frame.max() - rgb_frame.min())
        
        axes[1, 0].imshow(rgb_frame, aspect='auto', origin='lower')
        axes[1, 0].set_title('Combined RGB (Magma Colormap)')
        axes[1, 0].set_xlabel('Time Bins')
        axes[1, 0].set_ylabel('Mel Frequency Bands')
        
        # Frequency energy distribution
        freq_profile = np.mean(frame, axis=(0, 2))  # Average across channels and time
        axes[1, 1].plot(freq_profile, range(len(freq_profile)), 'b-', linewidth=2)
        axes[1, 1].fill_betweenx(range(len(freq_profile)), freq_profile, alpha=0.3)
        axes[1, 1].set_xlabel('Average Energy')
        axes[1, 1].set_ylabel('Mel Frequency Band')
        axes[1, 1].set_title('Energy Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'mel_spectrogram_participant_{participant_id}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Show temporal sequence if multiple frames
        if seq_len > 1:
            print(f"\n🕐 TEMPORAL SEQUENCE (showing up to 8 frames)")
            
            num_frames_to_show = min(8, seq_len)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'Temporal Evolution - Participant {participant_id} - {label_text}', 
                         fontsize=16, fontweight='bold')
            
            for i in range(num_frames_to_show):
                row, col = i // 4, i % 4
                frame = mel_data[i]
                rgb_frame = np.transpose(frame, (1, 2, 0))
                rgb_frame = (rgb_frame - rgb_frame.min()) / (rgb_frame.max() - rgb_frame.min())
                
                axes[row, col].imshow(rgb_frame, aspect='auto', origin='lower')
                axes[row, col].set_title(f'Frame {i} (t≈{i}s)')
                axes[row, col].set_xlabel('Time Bins')
                axes[row, col].set_ylabel('Mel Bands')
            
            # Hide unused subplots
            for i in range(num_frames_to_show, 8):
                row, col = i // 4, i % 4
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'temporal_sequence_participant_{participant_id}.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    return mel_data


def main():
    """Main function to visualize mel spectrograms."""
    print("🎵 MEL SPECTROGRAM VISUALIZATION FOR DEPRESSION DETECTION 🧠")
    print("=" * 70)
    
    print("""
🎯 WHAT YOU'RE LOOKING AT:
• Mel spectrograms converted to RGB images (AVEC 2019 approach)
• Each pixel represents energy at a specific frequency and time
• Brighter colors = more energy at that frequency
• Your CNN+GRU model processes these as "images" with temporal sequences

🔍 KEY FEATURES TO UNDERSTAND:
• Lower frequencies (bottom): Fundamental voice frequencies, prosody
• Higher frequencies (top): Voice quality, consonants, fricatives  
• Horizontal patterns: Temporal speech dynamics
• Vertical patterns: Harmonic structure

🏥 DEPRESSION MARKERS:
• Flat affect: More uniform energy distribution
• Reduced prosody: Less variation in lower frequencies
• Speech timing: Gaps and pauses (dark regions)
• Voice quality changes: Different high-frequency patterns
    """)
    
    # Try to find available participants
    data_path = Path("data/ExtendedDAIC")
    mel_path = data_path / "mel_spectrograms_avec2019"
    
    if not mel_path.exists():
        print("❌ Mel spectrogram directory not found!")
        return
    
    # Get available participants
    available_participants = []
    for participant_dir in mel_path.iterdir():
        if participant_dir.is_dir():
            participant_id = int(participant_dir.name)
            mel_file = participant_dir / f"{participant_id}_P.npy"
            if mel_file.exists():
                available_participants.append(participant_id)
    
    if not available_participants:
        print("❌ No mel spectrogram files found!")
        return
    
    print(f"✅ Found {len(available_participants)} participants with mel spectrograms")
    
    # Load label data to find examples from each class
    try:
        train_df = pd.read_csv(data_path / "train.csv")
        dev_df = pd.read_csv(data_path / "dev.csv")
        test_df = pd.read_csv(data_path / "test.csv")
        all_df = pd.concat([train_df, dev_df, test_df])
        
        # Find participants from each class
        available_df = all_df[all_df['Participant_ID'].isin(available_participants)]
        
        depressed_participants = available_df[available_df['PHQ_Binary'] == 1]['Participant_ID'].tolist()
        control_participants = available_df[available_df['PHQ_Binary'] == 0]['Participant_ID'].tolist()
        
        print(f"📊 Available data: {len(depressed_participants)} depressed, {len(control_participants)} control")
        
        # Visualize examples from each class
        if depressed_participants:
            print(f"\n🔍 ANALYZING DEPRESSED PARTICIPANT: {depressed_participants[0]}")
            load_and_visualize_mel_spectrogram(depressed_participants[0])
        
        if control_participants:
            print(f"\n🔍 ANALYZING CONTROL PARTICIPANT: {control_participants[0]}")
            load_and_visualize_mel_spectrogram(control_participants[0])
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Just show first available participant
        print(f"\n🔍 ANALYZING PARTICIPANT: {available_participants[0]}")
        load_and_visualize_mel_spectrogram(available_participants[0])
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("💡 You now understand how audio becomes 'images' for your CNN model!")
    print("🎯 Ready to explain to your supervisor how mel spectrograms work!")


if __name__ == "__main__":
    main() 