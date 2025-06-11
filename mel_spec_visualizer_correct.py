#!/usr/bin/env python3
"""
Corrected Mel Spectrogram Visualizer for Depression Detection
Data format: (3, width, height) = (channels, time_bins, mel_bands)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def visualize_mel_spectrogram(participant_id, data_path="data/ExtendedDAIC"):
    """Visualize mel spectrogram for a specific participant."""
    
    # Load the data
    mel_file = Path(data_path) / "mel_spectrograms_avec2019" / str(participant_id) / f"{participant_id}_P.npy"
    
    if not mel_file.exists():
        print(f"❌ File not found for participant {participant_id}")
        return None
    
    mel_data = np.load(mel_file)
    
    # Load label information
    try:
        train_df = pd.read_csv(Path(data_path) / "train.csv")
        dev_df = pd.read_csv(Path(data_path) / "dev.csv") 
        test_df = pd.read_csv(Path(data_path) / "test.csv")
        all_df = pd.concat([train_df, dev_df, test_df])
        
        participant_data = all_df[all_df['Participant_ID'] == participant_id]
        if len(participant_data) > 0:
            label = participant_data['PHQ_Binary'].iloc[0]
            label_text = "🔴 DEPRESSED" if label == 1 else "🟢 NOT DEPRESSED"
        else:
            label_text = "❓ UNKNOWN"
    except:
        label_text = "❓ UNKNOWN"
    
    print(f"\n🎵 PARTICIPANT {participant_id} - {label_text}")
    print("=" * 60)
    print(f"📊 Data Shape: {mel_data.shape}")
    print(f"📈 Value Range: [{mel_data.min():.3f}, {mel_data.max():.3f}]")
    
    # Understanding the data structure
    channels, time_bins, mel_bands = mel_data.shape
    print(f"📐 Structure:")
    print(f"  • Channels: {channels} (RGB from magma colormap)")
    print(f"  • Time bins: {time_bins} (entire conversation)")
    print(f"  • Mel bands: {mel_bands} (frequency resolution)")
    print(f"  • Estimated duration: ~{time_bins * 0.025:.1f} seconds")  # Assuming ~25ms per time bin
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Main title
    fig.suptitle(f'Mel Spectrogram Analysis: Participant {participant_id} - {label_text}', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Individual RGB channels (top row)
    channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
    channel_colors = ['Reds', 'Greens', 'Blues']
    
    for i in range(3):
        ax = plt.subplot(3, 3, i + 1)
        
        # Transpose from (time, freq) to (freq, time) for proper orientation
        channel_data = mel_data[i].T  # Now (mel_bands, time_bins)
        
        im = ax.imshow(channel_data, aspect='auto', origin='lower', 
                      cmap=channel_colors[i], interpolation='nearest')
        ax.set_title(f'{channel_names[i]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Bins →')
        ax.set_ylabel('Mel Frequency Bands ↑')
        
        # Add frequency labels
        ax.set_yticks([0, mel_bands//4, mel_bands//2, 3*mel_bands//4, mel_bands-1])
        ax.set_yticklabels(['50 Hz', '~500 Hz', '~1.5 kHz', '~4 kHz', '8 kHz'])
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 2. Combined RGB visualization (middle left)
    ax = plt.subplot(3, 3, 4)
    # Convert to proper RGB format: (height, width, channels)
    rgb_image = np.transpose(mel_data, (2, 1, 0))  # (mel_bands, time_bins, channels)
    
    # Normalize for display
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    
    ax.imshow(rgb_image, aspect='auto', origin='lower', interpolation='nearest')
    ax.set_title('Combined RGB\n(What CNN sees)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Bins →')
    ax.set_ylabel('Mel Frequency Bands ↑')
    ax.set_yticks([0, mel_bands//4, mel_bands//2, 3*mel_bands//4, mel_bands-1])
    ax.set_yticklabels(['50 Hz', '~500 Hz', '~1.5 kHz', '~4 kHz', '8 kHz'])
    
    # 3. Average energy over time (middle center)
    ax = plt.subplot(3, 3, 5)
    # Average energy across frequency bands for each time step
    temporal_energy = np.mean(mel_data, axis=(0, 2))  # Average over channels and mel bands
    time_axis = np.arange(len(temporal_energy)) * 0.025  # Convert to seconds
    
    ax.plot(time_axis, temporal_energy, 'b-', linewidth=2, alpha=0.8)
    ax.fill_between(time_axis, temporal_energy, alpha=0.3)
    ax.set_title('Energy Over Time\n(Speech Activity)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Average Energy')
    ax.grid(True, alpha=0.3)
    
    # 4. Average energy per frequency band (middle right)
    ax = plt.subplot(3, 3, 6)
    # Average energy across time and channels for each frequency band
    freq_energy = np.mean(mel_data, axis=(0, 1))  # Average over channels and time
    mel_axis = np.arange(len(freq_energy))
    
    ax.barh(mel_axis, freq_energy, alpha=0.7, color='purple')
    ax.set_title('Frequency Distribution\n(Voice Characteristics)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Average Energy')
    ax.set_ylabel('Mel Frequency Band')
    ax.set_yticks([0, mel_bands//4, mel_bands//2, 3*mel_bands//4, mel_bands-1])
    ax.set_yticklabels(['50 Hz', '~500 Hz', '~1.5 kHz', '~4 kHz', '8 kHz'])
    ax.grid(True, alpha=0.3)
    
    # 5. Spectral centroid over time (bottom left)
    ax = plt.subplot(3, 3, 7)
    # Calculate spectral centroid (brightness measure)
    freq_weights = np.arange(mel_bands)
    spectral_centroids = []
    for t in range(time_bins):
        spectrum = np.mean(mel_data[:, t, :], axis=0)  # Average over channels
        if spectrum.sum() > 0:
            centroid = np.sum(freq_weights * spectrum) / spectrum.sum()
        else:
            centroid = 0
        spectral_centroids.append(centroid)
    
    ax.plot(time_axis, spectral_centroids, 'g-', linewidth=2, alpha=0.8)
    ax.set_title('Spectral Centroid\n(Voice Brightness)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency Band')
    ax.grid(True, alpha=0.3)
    
    # 6. Energy variance over time (bottom center)
    ax = plt.subplot(3, 3, 8)
    # Calculate energy variance across frequency bands for each time step
    energy_variance = []
    for t in range(time_bins):
        spectrum = np.mean(mel_data[:, t, :], axis=0)
        energy_variance.append(np.var(spectrum))
    
    ax.plot(time_axis, energy_variance, 'r-', linewidth=2, alpha=0.8)
    ax.fill_between(time_axis, energy_variance, alpha=0.3, color='red')
    ax.set_title('Spectral Variance\n(Voice Complexity)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Energy Variance')
    ax.grid(True, alpha=0.3)
    
    # 7. Summary statistics (bottom right)
    ax = plt.subplot(3, 3, 9)
    ax.axis('off')
    
    # Calculate summary statistics
    total_energy = np.mean(mel_data)
    max_energy = np.max(mel_data)
    energy_std = np.std(mel_data)
    silence_ratio = np.mean(temporal_energy < 0.1)  # Rough silence detection
    
    stats_text = f"""
📊 SUMMARY STATISTICS:
    
🔊 Average Energy: {total_energy:.3f}
📈 Peak Energy: {max_energy:.3f}
📉 Energy Std Dev: {energy_std:.3f}
🔇 Silence Ratio: {silence_ratio:.1%}

🎯 DEPRESSION MARKERS:
• Flat affect: Low energy variance
• Reduced prosody: Low spectral variance
• Speech patterns: Silence distribution
• Voice quality: Frequency distribution

🤖 CNN PROCESSING:
• Each pixel = energy at freq/time
• Patterns detected like visual features
• RGB channels capture spectral details
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the visualization
    filename = f'mel_analysis_participant_{participant_id}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"💾 Saved visualization as: {filename}")
    
    plt.show()
    
    return mel_data


def main():
    """Main visualization function."""
    print("🎵 MEL SPECTROGRAM DEEP DIVE FOR DEPRESSION DETECTION 🧠")
    print("=" * 70)
    
    print("""
🎯 UNDERSTANDING MEL SPECTROGRAMS AS IMAGES:

📊 Data Structure: (3, 256, 128)
  • 3 channels: RGB representation (magma colormap)
  • 256 time bins: Temporal progression of conversation
  • 128 mel bands: Frequency content (50 Hz to 8 kHz)

🔍 What CNNs See:
  • Horizontal patterns: Speech rhythm, pauses, timing
  • Vertical patterns: Harmonic structure, voice quality
  • Color intensity: Energy at specific frequency/time points
  • Texture patterns: Prosodic features, emotional markers

🏥 Depression Detection Clues:
  • Flat affect: More uniform energy distribution
  • Reduced prosody: Less frequency modulation
  • Speech timing: Different pause patterns
  • Voice quality: Changes in harmonic structure
    """)
    
    # Find available participants
    data_path = Path("data/ExtendedDAIC")
    mel_path = data_path / "mel_spectrograms_avec2019"
    
    if not mel_path.exists():
        print("❌ Mel spectrogram directory not found!")
        return
    
    # Get available participants
    available_participants = []
    for participant_dir in mel_path.iterdir():
        if participant_dir.is_dir() and participant_dir.name.isdigit():
            participant_id = int(participant_dir.name)
            mel_file = participant_dir / f"{participant_id}_P.npy"
            if mel_file.exists():
                available_participants.append(participant_id)
    
    available_participants.sort()
    print(f"✅ Found {len(available_participants)} participants")
    
    # Load labels and find examples from each class
    try:
        train_df = pd.read_csv(data_path / "train.csv")
        dev_df = pd.read_csv(data_path / "dev.csv")
        test_df = pd.read_csv(data_path / "test.csv")
        all_df = pd.concat([train_df, dev_df, test_df])
        
        available_df = all_df[all_df['Participant_ID'].isin(available_participants)]
        depressed = available_df[available_df['PHQ_Binary'] == 1]['Participant_ID'].tolist()
        control = available_df[available_df['PHQ_Binary'] == 0]['Participant_ID'].tolist()
        
        print(f"📈 Available: {len(depressed)} depressed, {len(control)} control participants")
        
        # Visualize one from each class
        if depressed:
            print(f"\n🔍 ANALYZING DEPRESSED PARTICIPANT")
            visualize_mel_spectrogram(depressed[0])
        
        if control:
            print(f"\n🔍 ANALYZING CONTROL PARTICIPANT")
            visualize_mel_spectrogram(control[0])
            
    except Exception as e:
        print(f"⚠️ Error loading labels: {e}")
        if available_participants:
            print(f"\n🔍 ANALYZING PARTICIPANT {available_participants[0]}")
            visualize_mel_spectrogram(available_participants[0])
    
    print("\n" + "=" * 70)
    print("🎉 VISUALIZATION COMPLETE!")
    print("✨ You now see how audio becomes 'images' for CNN analysis!")
    print("🎯 Ready to impress your supervisor with deep mel spectrogram knowledge!")
    print("\n💡 Key Takeaway: The genius of AVEC 2019 was treating audio as computer vision!")


if __name__ == "__main__":
    main() 