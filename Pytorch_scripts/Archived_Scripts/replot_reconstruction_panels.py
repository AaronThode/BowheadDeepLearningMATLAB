#!/usr/bin/env python3
"""
Regenerate reconstruction panels from existing trained model without retraining.
Uses the save_reconstruction_panels function with proper axis labels, filenames, and metadata.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import ConcatDataset
import math

# Import the autoencoder architecture and helper functions
from Autoencoder_v02_20251118 import (
    ImprovedAutoencoder, 
    SNRDataset,
    select_samples_for_outputs,
    save_reconstruction_panels,
    match_shape_center
)

# Model configurations
MODELS = {
    'v06': {
        'name': 'MostlyManual_50K',
        'results_dir': 'Autoencoder_v06_100E_32LD_MostlyManual_50K_Date20251121-170008.dir',
        'datasets': ['/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir'],
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'epochs': 100,
        'extra_conv': False
    },
    'v07': {
        'name': 'AutoWithAirguns_50K',
        'results_dir': 'Autoencoder_v07_100E_32LD_AutoWithAirguns_50K_Date20251123-001830.dir',
        'datasets': ['/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir'],
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'epochs': 100,
        'extra_conv': False
    },
    'v08': {
        'name': 'CombinedDatasets_100K',
        'results_dir': 'Autoencoder_v08_100E_32LD_CombinedDatasets_100K_Date20251125-171340.dir',
        'datasets': ['/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir',
                    '/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir'],
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'epochs': 100,
        'extra_conv': False
    },
    'v09': {
        'name': 'CombinedDatasets_100K',
        'results_dir': 'Autoencoder_v09_100E_32LD_CombinedDatasets_100K_Date20251209-122650.dir',
        'datasets': ['/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir',
                    '/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/Unsupervised_database_MostlyManual.dir'],
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 16,
        'base_channels': 64,
        'epochs': 100,
        'extra_conv': False
    }
}


def replot_panels(model_version='v08', num_samples=30, seed=42, show_error=True):
    """
    Regenerate reconstruction panels from existing model.
    
    Args:
        model_version: Which model to use ('v06', 'v07', 'v08', 'v09')
        num_samples: Number of samples to plot
        seed: Random seed for sample selection
        show_error: Whether to show error plots
    """
    print("=" * 70)
    print(f"REPLOTTING RECONSTRUCTION PANELS: {model_version}")
    print("=" * 70)
    
    config = MODELS[model_version]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)
    results_dir = os.path.join(base_dir, 'results', config['results_dir'])
    
    # Load model
    print(f"\nLoading trained model...")
    model_path = os.path.join(results_dir, config['model_file'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = ImprovedAutoencoder(
        nrow=121,
        ncol=104,
        latent_dim=config['latent_dim'],
        base_channels=config['base_channels'],
        extra_conv=config['extra_conv']
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"  ✓ Model loaded from: {model_path}")
    print(f"  ✓ Architecture: latent_dim={config['latent_dim']}, channels={config['base_channels']}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    if len(config['datasets']) == 1:
        dataset = SNRDataset(config['datasets'][0], normalize=True, seed=seed, show_summary=True)
        dataset_label = os.path.basename(config['datasets'][0].rstrip('/'))
    else:
        datasets = []
        for i, dir_path in enumerate(config['datasets'], 1):
            print(f"  [{i}] {os.path.basename(dir_path)}")
            ds = SNRDataset(dir_path, normalize=True, seed=seed, show_summary=False)
            datasets.append(ds)
            print(f"      Loaded {len(ds)} samples")
        dataset = ConcatDataset(datasets)
        dataset_label = config['name']
        print(f"\nTotal combined samples: {len(dataset)}")
    
    # Select samples
    print(f"\nSelecting {num_samples} random samples...")
    panel_samples, panel_filenames = select_samples_for_outputs(dataset, num_samples, seed)
    print(f"  ✓ Selected {panel_samples.shape[0]} samples")
    
    # Generate panels
    print(f"\nGenerating reconstruction panels...")
    panels_written = save_reconstruction_panels(
        model=model,
        samples=panel_samples,
        output_dir=results_dir,
        target_hw=(121, 104),
        base_name="recon_panel",
        dataset_label=dataset_label,
        filenames=panel_filenames,
        show_error=show_error,
        epochs=config['epochs'],
        latent_dim=config['latent_dim'],
        channels=config['base_channels']
    )
    
    print(f"\n{'=' * 70}")
    print(f"COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Panels written: {panels_written}")
    print(f"Output directory: {results_dir}")
    print(f"Files: recon_panel_001.jpg, recon_panel_002.jpg, ...")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate reconstruction panels from existing trained model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='v08',
        choices=['v06', 'v07', 'v08', 'v09'],
        help='Model version to use (default: v08)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=30,
        help='Number of samples to plot (default: 30)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample selection (default: 42)'
    )
    parser.add_argument(
        '--no-error',
        action='store_true',
        help='Disable error plots (show only input and reconstruction)'
    )
    
    args = parser.parse_args()
    
    replot_panels(
        model_version=args.model,
        num_samples=args.samples,
        seed=args.seed,
        show_error=not args.no_error
    )


if __name__ == '__main__':
    main()
