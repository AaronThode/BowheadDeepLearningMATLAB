#!/usr/bin/env python3
"""
Replot reconstruction comparison with frequency/time axes.
Loads existing reconstruction_data.mat and regenerates the figure.

USAGE:
    python replot_reconstructions.py <results_dir>
    python replot_reconstructions.py /Autoencoder_v04_32LD_Balanced_Date20251117-185303.dir
    python replot_reconstructions.py /Autoencoder_v04_32LD_HighAirguns_Date20251118-114536.dir
    python replot_reconstructions.py /Autoencoder_v04_32LD_MostlyManual_Date20251118-085305.dir


"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from scipy.io import loadmat

def replot_reconstructions(results_dir, show_error=False):
    """Replot reconstructions with proper frequency/time axes."""
    
    # Load saved data
    data_path = os.path.join(results_dir, 'reconstruction_data.mat')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No reconstruction_data.mat found in {results_dir}")
    
    # Load timing log to get training parameters
    timing_log_path = os.path.join(results_dir, 'timing_log.txt')
    epochs = None
    latent_dim = None
    channels = None
    dataset_label = None
    
    if os.path.exists(timing_log_path):
        with open(timing_log_path, 'r') as f:
            for line in f:
                if 'Epochs:' in line:
                    epochs = int(line.split(':')[1].strip())
                elif 'Latent dim:' in line:
                    latent_dim = int(line.split(':')[1].strip())
                elif 'Channels:' in line:
                    channels = int(line.split(':')[1].strip())
                elif 'Dataset:' in line:
                    dataset_label = line.split(':')[1].strip()
    
    print(f"Loading data from: {data_path}")
    data = loadmat(data_path)
    data_np = data['spectrograms']
    recon_np = data['reconstructions']
    
    print(f"Loaded {data_np.shape[0]} samples")
    if epochs:
        print(f"Training config: epochs={epochs}, latent_dim={latent_dim}, channels={channels}")
    
    # Plot parameters
    vmin_data = data_np.min()
    vmax_data = data_np.max()
    cols = min(10, data_np.shape[0])
    n_rows = 3 if show_error else 2
    
    nrow, ncol = data_np.shape[1], data_np.shape[2]
    
    # Frequency axis parameters: 121 rows, start=27.3438 Hz, step=3.9062 Hz, max=500 Hz
    freq_start = 10
    freq_step = 5
    freq_ticks = np.arange(0, nrow, 30)  # Show tick every 30 rows
    freq_labels = [f'{freq_start + tick * freq_step:.0f}' for tick in freq_ticks]
    
    # Time axis parameters: 104 columns, each 0.026 seconds, total=2.7 seconds
    time_step = 0.026
    total_time = ncol * time_step
    
    # Major ticks: every 1 second (with labels)
    major_tick_interval = 1.0
    num_major_ticks = int(total_time / major_tick_interval) + 1
    major_tick_values = np.arange(num_major_ticks) * major_tick_interval
    major_ticks = major_tick_values / time_step  # Convert seconds to column indices
    major_labels = [f'{t:.1f}' for t in major_tick_values]
    
    # Minor ticks: every 0.5 seconds (no labels)
    minor_tick_interval = 0.5
    num_minor_ticks = int(total_time / minor_tick_interval) + 1
    minor_tick_values = np.arange(num_minor_ticks) * minor_tick_interval
    minor_ticks = minor_tick_values / time_step
    
    # Create figure
    fig, axes = plt.subplots(n_rows, cols, figsize=(10, 6 if n_rows == 2 else 9))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i in range(cols):
        # Input row
        axes[0, i].imshow(data_np[i], cmap='viridis', origin='lower', aspect='auto', 
                         vmin=vmin_data, vmax=vmax_data)
        axes[0, i].set_title(f'Input {i+1}')
        if i == 0:  # Only show axes on first column
            axes[0, i].set_yticks(freq_ticks)
            axes[0, i].set_yticklabels(freq_labels)
            axes[0, i].set_ylabel('Frequency (Hz)', fontsize=10)
            axes[0, i].set_xticks(major_ticks)
            axes[0, i].set_xticklabels(major_labels)
            axes[0, i].set_xticks(minor_ticks, minor=True)
            axes[0, i].set_xlabel('Time (s)', fontsize=10)
            axes[0, i].tick_params(axis='x', which='minor', length=3)
        else:
            axes[0, i].axis('off')
        
        # Reconstruction row
        axes[1, i].imshow(recon_np[i], cmap='viridis', origin='lower', aspect='auto', 
                         vmin=vmin_data, vmax=vmax_data)
        axes[1, i].set_title(f'Recon {i+1}')
        if i == 0:  # Only show axes on first column
            axes[1, i].set_yticks(freq_ticks)
            axes[1, i].set_yticklabels(freq_labels)
            axes[1, i].set_ylabel('Frequency (Hz)', fontsize=10)
            axes[1, i].set_xticks(major_ticks)
            axes[1, i].set_xticklabels(major_labels)
            axes[1, i].set_xticks(minor_ticks, minor=True)
            axes[1, i].set_xlabel('Time (s)', fontsize=10)
            axes[1, i].tick_params(axis='x', which='minor', length=3)
        else:
            axes[1, i].axis('off')
        
        # Error row (if enabled)
        if show_error:
            diff = np.abs(data_np[i] - recon_np[i])
            axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
            axes[2, i].set_title(f'Error {i+1}')
            if i == 0:  # Only show axes on first column
                axes[2, i].set_yticks(freq_ticks)
                axes[2, i].set_yticklabels(freq_labels)
                axes[2, i].set_ylabel('Frequency (Hz)', fontsize=10)
                axes[2, i].set_xticks(major_ticks)
                axes[2, i].set_xticklabels(major_labels)
                axes[2, i].set_xticks(minor_ticks, minor=True)
                axes[2, i].set_xlabel('Time (s)', fontsize=10)
                axes[2, i].tick_params(axis='x', which='minor', length=3)
            else:
                axes[2, i].axis('off')
    
    # Add title above center columns (around Input 5-6 position)
    title_text = f'Autoencoder Reconstructions (epochs={epochs}, latent_dim={latent_dim}, channels={channels})' if (epochs and latent_dim and channels) else 'Autoencoder Reconstructions'
    plt.figtext(0.5, 0.98, title_text, ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Add dataset label in bottom right corner
    if dataset_label:
        plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', 
                   ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save with new name
    output_path = os.path.join(results_dir, 'reconstructions_with_axes.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved replotted figure to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replot_reconstructions.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    replot_reconstructions(results_dir, show_error=False)
