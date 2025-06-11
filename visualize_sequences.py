#!/usr/bin/env python3
"""
Visualization script to check the custom preprocessing output.
Shows the first 10 spectrograms from patient 300's sequence.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_spectrogram_sequence(patient_id='300', num_spectrograms=10):
    """
    Visualize the first N spectrograms from a patient's sequence.
    
    Args:
        patient_id: Patient ID to visualize
        num_spectrograms: Number of spectrograms to show
    """
    # Load the sequence data
    data_file = Path(f'data/ExtendedDAIC/mel_spectrograms_avec2019/{patient_id}/{patient_id}_P.npy')
    
    if not data_file.exists():
        print(f"Error: File {data_file} not found!")
        print("Make sure you've run the custom preprocessing script first.")
        return
    
    # Load the data
    spectrograms = np.load(data_file)
    
    print(f"Loaded sequence data for patient {patient_id}")
    print(f"Shape: {spectrograms.shape}")
    print(f"Format: (num_windows={spectrograms.shape[0]}, channels={spectrograms.shape[1]}, height={spectrograms.shape[2]}, width={spectrograms.shape[3]})")
    print(f"Data type: {spectrograms.dtype}")
    print(f"Value range: [{spectrograms.min()}, {spectrograms.max()}]")
    
    # Limit to available spectrograms
    num_to_show = min(num_spectrograms, spectrograms.shape[0])
    
    # Create subplot grid
    cols = 5
    rows = (num_to_show + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_to_show):
        row = i // cols
        col = i % cols
        
        # Get the spectrogram (channels, height, width) -> (height, width, channels)
        spectrogram = spectrograms[i].transpose(1, 2, 0)
        
        # Display the spectrogram
        axes[row, col].imshow(spectrogram, aspect='auto', origin='lower')
        axes[row, col].set_title(f'Window {i+1}\n(t={i:.1f}-{i+4:.1f}s)')
        axes[row, col].set_xlabel('Time bins')
        axes[row, col].set_ylabel('Frequency bins')
        
        # Add time and frequency labels
        if col == 0:  # Only on leftmost plots
            axes[row, col].set_ylabel('Mel bins (50-8000 Hz)')
        if row == rows-1:  # Only on bottom plots
            axes[row, col].set_xlabel('Time bins (4s window)')
    
    # Hide unused subplots
    for i in range(num_to_show, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Patient {patient_id}: First {num_to_show} Mel Spectrograms\n4-second windows with 1-second hop', 
                 fontsize=14, y=1.02)
    
    # Save the visualization
    output_file = f'patient_{patient_id}_spectrograms_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_file}")
    
    plt.show()


def compare_with_original():
    """
    Compare the new sequence format with the original single spectrogram.
    """
    patient_id = '300'
    
    # Load original single spectrogram
    original_file = Path(f'data/ExtendedDAIC/mel_spectrograms_avec2019/{patient_id}/{patient_id}_P.npy')
    
    # Check if we have the original format or new format
    if original_file.exists():
        data = np.load(original_file)
        
        print(f"Current data shape: {data.shape}")
        
        if len(data.shape) == 3:
            print("This appears to be the OLD format (single spectrogram)")
            print("You need to run the custom preprocessing script first!")
            return
        elif len(data.shape) == 4:
            print("This is the NEW format (sequence of spectrograms)")
            print(f"Number of windows: {data.shape[0]}")
            print(f"Expected windows for ~145s audio: {int((145-4)/1)+1} = 142")
            
            # Show statistics
            print(f"\nSequence statistics:")
            print(f"- Total windows: {data.shape[0]}")
            print(f"- Window size: {data.shape[2]}×{data.shape[3]} (height×width)")
            print(f"- Channels: {data.shape[1]} (RGB)")
            print(f"- Coverage: {data.shape[0] * 1:.1f}s with 4s windows + 3s final window")
            print(f"- Original audio was ~144.8s")
    else:
        print(f"File {original_file} not found!")


def main():
    print("=== Mel Spectrogram Sequence Visualization ===\n")
    
    # First, check the data format
    compare_with_original()
    print("\n" + "="*50 + "\n")
    
    # Then visualize the spectrograms
    visualize_spectrogram_sequence(patient_id='300', num_spectrograms=10)


if __name__ == "__main__":
    main() 