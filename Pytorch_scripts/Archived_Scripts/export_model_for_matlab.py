#!/usr/bin/env python3
"""
Export PyTorch Autoencoder to TorchScript for MATLAB

This script exports a trained PyTorch model to TorchScript (.pt) format,
which can be imported and used directly in MATLAB.

Usage:
    python export_model_for_matlab.py --model v08
    python export_model_for_matlab.py --model v09 --output-name autoencoder_v09.pt
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat, savemat

# Import the autoencoder architecture
from Autoencoder_v02_20251118 import ImprovedAutoencoder

# Model configurations
MODELS = {
    'v06': {
        'name': 'MostlyManual_50K',
        'results_dir': 'Autoencoder_v06_100E_32LD_MostlyManual_50K_Date20251121-170008.dir',
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'extra_conv': False
    },
    'v07': {
        'name': 'AutoWithAirguns_50K',
        'results_dir': 'Autoencoder_v07_100E_32LD_AutoWithAirguns_50K_Date20251123-001830.dir',
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'extra_conv': False
    },
    'v08': {
        'name': 'CombinedDatasets_100K',
        'results_dir': 'Autoencoder_v08_100E_32LD_CombinedDatasets_100K_Date20251125-171340.dir',
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 32,
        'base_channels': 64,
        'extra_conv': False
    },
    'v09': {
        'name': 'CombinedDatasets_100K',
        'results_dir': 'Autoencoder_v09_100E_32LD_CombinedDatasets_100K_Date20251209-122650.dir',
        'model_file': 'autoencoder_clean.pth',
        'latent_dim': 16,
        'base_channels': 64,
        'extra_conv': False
    }
}


def export_to_torchscript(model_version='v08', output_name=None):
    """
    Export trained model to TorchScript format for MATLAB.
    
    Args:
        model_version: Which model to export ('v06', 'v07', 'v08', 'v09')
        output_name: Optional custom output filename
    """
    print("=" * 70)
    print(f"EXPORTING MODEL TO TORCHSCRIPT: {model_version}")
    print("=" * 70)
    
    config = MODELS[model_version]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)
    results_dir = os.path.join(base_dir, 'results', config['results_dir'])
    
    # Load model
    print(f"\nStep 1: Loading trained model...")
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
    
    # Create example input for tracing
    print(f"\nStep 2: Creating TorchScript traced model...")
    example_input = torch.randn(1, 1, 121, 104)  # Batch=1, Channels=1, Height=121, Width=104
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    print(f"  ✓ Model traced successfully")
    
    # Save TorchScript model
    print(f"\nStep 3: Saving TorchScript model...")
    if output_name is None:
        output_name = f"autoencoder_{model_version}_torchscript.pt"
    
    output_path = os.path.join(results_dir, output_name)
    torch.jit.save(traced_model, output_path)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ File size: {file_size_mb:.1f} MB")
    
    # Test the exported model
    print(f"\nStep 4: Testing exported model...")
    loaded_traced = torch.jit.load(output_path)
    with torch.no_grad():
        test_output, test_latent = loaded_traced(example_input)
    print(f"  ✓ Model test passed")
    print(f"  ✓ Input shape:  {example_input.shape}")
    print(f"  ✓ Output shape: {test_output.shape}")
    print(f"  ✓ Latent shape: {test_latent.shape}")
    
    # Create MATLAB helper info file
    print(f"\nStep 5: Creating MATLAB helper file...")
    matlab_info = {
        'model_version': model_version,
        'model_name': config['name'],
        'input_shape': np.array([1, 1, 121, 104]),
        'output_shape': np.array(list(test_output.shape)),
        'latent_dim': config['latent_dim'],
        'base_channels': config['base_channels'],
        'torchscript_file': output_name,
        'instructions': 'Use PyTorch interface in MATLAB to load this model'
    }
    
    info_path = os.path.join(results_dir, f"matlab_model_info_{model_version}.mat")
    savemat(info_path, matlab_info)
    print(f"  ✓ Saved info to: {info_path}")
    
    # Create MATLAB usage example
    matlab_example = f"""% MATLAB Example: Load and Use PyTorch Model
% ============================================================================

% 1. Load the TorchScript model
model = torchModel('{output_path}');

% 2. Prepare your input spectrogram (121 x 104 matrix)
% Example: Load from your .mat file
data = load('your_spectrogram.mat');
input_spec = data.SNR_gram;  % Should be 121 x 104

% 3. Normalize input (0 to 1 range)
input_spec = (input_spec - min(input_spec(:))) / (max(input_spec(:)) - min(input_spec(:)));

% 4. Convert to dlarray (Deep Learning Toolbox required)
input_dlarray = dlarray(single(input_spec), 'SSCB');  % Spatial, Spatial, Channel, Batch
input_dlarray = reshape(input_dlarray, [121, 104, 1, 1]);  % Add channel and batch dims

% 5. Run inference
[reconstruction, latent] = predict(model, input_dlarray);

% 6. Extract results
recon_spec = extractdata(reconstruction);
recon_spec = squeeze(recon_spec);  % Remove extra dimensions (121 x 104)
latent_vector = extractdata(latent);
latent_vector = squeeze(latent_vector);  % {config['latent_dim']}D vector

% 7. Visualize
figure;
subplot(1,2,1); imagesc(input_spec); colorbar; title('Original');
subplot(1,2,2); imagesc(recon_spec); colorbar; title('Reconstruction');

% Note: Requires MATLAB R2020b or later with Deep Learning Toolbox
% and Deep Learning Toolbox Converter for PyTorch Models
"""
    
    example_path = os.path.join(results_dir, f"matlab_usage_example_{model_version}.m")
    with open(example_path, 'w') as f:
        f.write(matlab_example)
    print(f"  ✓ Saved MATLAB example to: {example_path}")
    
    print("\n" + "=" * 70)
    print("EXPORT COMPLETE!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. TorchScript model: {output_name}")
    print(f"  2. Model info (.mat): matlab_model_info_{model_version}.mat")
    print(f"  3. MATLAB example:    matlab_usage_example_{model_version}.m")
    print(f"\nLocation: {results_dir}")
    print("\nTo use in MATLAB:")
    print(f"  1. Navigate to: {results_dir}")
    print(f"  2. Run: model = torchModel('{output_name}');")
    print(f"  3. See matlab_usage_example_{model_version}.m for full example")
    print("\nRequirements:")
    print("  - MATLAB R2020b or later")
    print("  - Deep Learning Toolbox")
    print("  - Deep Learning Toolbox Converter for PyTorch Models")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Export PyTorch autoencoder to TorchScript for MATLAB'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='v08',
        choices=['v06', 'v07', 'v08', 'v09'],
        help='Model version to export (default: v08)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Custom output filename (default: autoencoder_<version>_torchscript.pt)'
    )
    
    args = parser.parse_args()
    
    export_to_torchscript(
        model_version=args.model,
        output_name=args.output_name
    )


if __name__ == '__main__':
    main()
