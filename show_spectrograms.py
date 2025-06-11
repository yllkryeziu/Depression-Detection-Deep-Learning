#!/usr/bin/env python3
"""
Simple script to display individual spectrograms from the sequence.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def show_individual_spectrograms():
    """Show first 6 spectrograms in a 2x3 grid for clear viewing."""
    
    # Load patient 300 data
    data_file = Path('data/ExtendedDAIC/mel_spectrograms_avec2019/300/300_P.npy')
    spectrograms = np.load(data_file)
    
    print(f"Patient 300: {spectrograms.shape[0]} spectrograms available")
    
    # Show first 6 spectrograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i in range(6):
        row = i // 3
        col = i % 3
        
        # Get spectrogram and convert from (C,H,W) to (H,W,C)
        spec = spectrograms[i].transpose(1, 2, 0)
        
        # Display
        im = axes[row, col].imshow(spec, aspect='auto', origin='lower')
        axes[row, col].set_title(f'Window {i+1}: {i:.1f}s - {i+4:.1f}s\n4-second mel spectrogram', 
                                fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Time bins (160 bins = 4 seconds)')
        axes[row, col].set_ylabel('Mel frequency bins (128 bins, 50-8000 Hz)')
        
        # Add grid for better readability
        axes[row, col].grid(True, alpha=0.3)
        
        # Add colorbar to first plot
        if i == 0:
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.suptitle('Patient 300: First 6 Mel Spectrograms (4s windows, 1s hop)\nMagma colormap, 128 mel bins, 160 time bins', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Save
    plt.savefig('patient_300_individual_spectrograms.png', dpi=300, bbox_inches='tight')
    print("Saved: patient_300_individual_spectrograms.png")
    
    plt.show()


def analyze_temporal_progression():
    """Analyze how spectrograms change over time."""
    
    data_file = Path('data/ExtendedDAIC/mel_spectrograms_avec2019/300/300_P.npy')
    spectrograms = np.load(data_file)
    
    print(f"\n=== Temporal Analysis ===")
    print(f"Total spectrograms: {spectrograms.shape[0]}")
    print(f"Time coverage: 0s to {spectrograms.shape[0] + 3}s")
    print(f"Overlap: Each window overlaps 3s with the next")
    
    # Show spectrograms from different time periods
    indices = [0, 20, 40, 60, 80, 100]  # Different time points
    times = [(i, i+4) for i in indices]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, i in enumerate(indices):
        if i >= spectrograms.shape[0]:
            continue
            
        row = idx // 3
        col = idx % 3
        
        spec = spectrograms[i].transpose(1, 2, 0)
        
        axes[row, col].imshow(spec, aspect='auto', origin='lower')
        axes[row, col].set_title(f'Window {i+1}: {times[idx][0]}s-{times[idx][1]}s', 
                                fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Time bins')
        axes[row, col].set_ylabel('Mel bins')
    
    plt.tight_layout()
    plt.suptitle('Patient 300: Spectrograms from Different Time Periods\nShowing temporal progression through conversation', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig('patient_300_temporal_progression.png', dpi=300, bbox_inches='tight')
    print("Saved: patient_300_temporal_progression.png")
    
    plt.show()


if __name__ == "__main__":
    show_individual_spectrograms()
    analyze_temporal_progression() 