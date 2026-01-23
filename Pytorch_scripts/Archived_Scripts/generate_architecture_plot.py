#!/usr/bin/env python3
"""
Generate PlotNeuralNet LaTeX code for the Whale Call Autoencoder Architecture

Architecture (with base_channels=64, latent_dim=32):
Input: 121x104 spectrogram
Encoder: 
  - Conv2D(1→64) + BatchNorm + ReLU + MaxPool → 60x52
  - Conv2D(64→128) + BatchNorm + ReLU + MaxPool → 30x26
  - Conv2D(128→256) + BatchNorm + ReLU + MaxPool → 15x13
  - Flatten → 49920
  - Linear(49920→64) + ReLU
  - Linear(64→32) → LATENT SPACE
Decoder:
  - Linear(32→64) + ReLU
  - Linear(64→49920) + ReLU
  - Reshape → 256x15x13
  - ConvTranspose2D(256→128) + BatchNorm + ReLU → 30x26
  - ConvTranspose2D(128→64) + BatchNorm + ReLU → 60x52
  - ConvTranspose2D(64→1) → 121x104
Output: Reconstructed spectrogram
"""

import os

# ============================================================================
# CUSTOMIZABLE LAYER LABELS - EDIT THESE TO CHANGE LABELS IN THE DIAGRAM
# ============================================================================
LAYER_LABELS = {
    "input": "Input",
    "conv1": "Feature Extraction",
    "pool1": "Downsampling",
    "conv2": "Feature Extraction",
    "pool2": "Downsampling",
    "conv3": "Feature Extraction",
    "pool3": "Downsampling",
    "dense1": "Compression",
    "latent": "Compressed Representation",
    "dense2": "Expansion",
    "dense3": "Expansion",
    "deconv1": "Upsampling",
    "deconv2": "Upsampling",
    "output": "Output",
}

# Layer dimensions [height, width, channels] or [features] for dense layers
LAYER_DIMS = {
    "input": [121, 104, 1],
    "conv1": [60, 52, 64],
    "pool1": [60, 52, 64],
    "conv2": [30, 26, 128],
    "pool2": [30, 26, 128],
    "conv3": [15, 13, 256],
    "pool3": [15, 13, 256],
    "dense1": [64],
    "latent": [32],
    "dense2": [64],
    "dense3": [49920],
    "deconv1": [30, 26, 128],
    "deconv2": [60, 52, 64],
    "output": [121, 104, 1],
}

def format_dims(dims):
    """Format dimensions as [x, y, z] or [n] string."""
    if len(dims) == 1:
        return f"[{dims[0]}]"
    elif len(dims) == 3:
        return f"[{dims[0]}, {dims[1]}, {dims[2]}]"
    return str(dims)

def generate_label_node(name, yshift="-2.5cm"):
    """Generate LaTeX node for layer label with dimensions."""
    label = LAYER_LABELS.get(name, name)
    dims = format_dims(LAYER_DIMS.get(name, []))
    return f"\\node[below={yshift} of {name}, align=center] {{\\textbf{{{label}}}\\\\\\small {dims}}};"

# Create the LaTeX code for PlotNeuralNet
latex_code = r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{layers/}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d}

\def\ConvColor{rgb:blue,2;white,8}
\def\ConvReluColor{rgb:blue,3;white,7}
\def\PoolColor{rgb:blue,2;white,8}
\def\DenseColor{rgb:blue,2;white,8}
\def\LatentColor{rgb:blue,4;white,6}
\def\SoftmaxColor{rgb:blue,3;white,7}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ENCODER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input spectrogram (121x104x1)
\pic[shift={(0,0,0)}] at (0,0,0) {Box={
    name=input,
    caption=,
    fill=\ConvColor,
    height=50,
    width=2,
    depth=45}};

%% Conv Block 1: 64 filters
\pic[shift={(5,0,0)}] at (input-east) {RightBandedBox={
    name=conv1,
    caption=,
    fill=\ConvColor,
    bandfill=\ConvReluColor,
    height=45,
    width=8,
    depth=38}};

%% Pool 1
\pic[shift={(4,0,0)}] at (conv1-east) {Box={
    name=pool1,
    caption=,
    fill=\PoolColor,
    opacity=0.5,
    height=42,
    width=8,
    depth=35}};

%% Conv Block 2: 128 filters
\pic[shift={(5,0,0)}] at (pool1-east) {RightBandedBox={
    name=conv2,
    caption=,
    fill=\ConvColor,
    bandfill=\ConvReluColor,
    height=35,
    width=12,
    depth=28}};

%% Pool 2
\pic[shift={(4,0,0)}] at (conv2-east) {Box={
    name=pool2,
    caption=,
    fill=\PoolColor,
    opacity=0.5,
    height=32,
    width=12,
    depth=25}};

%% Conv Block 3: 256 filters
\pic[shift={(5,0,0)}] at (pool2-east) {RightBandedBox={
    name=conv3,
    caption=,
    fill=\ConvColor,
    bandfill=\ConvReluColor,
    height=25,
    width=16,
    depth=18}};

%% Pool 3
\pic[shift={(4,0,0)}] at (conv3-east) {Box={
    name=pool3,
    caption=,
    fill=\PoolColor,
    opacity=0.5,
    height=22,
    width=16,
    depth=15}};

%% Flatten indicator
\draw[connection] (pool3-east) -- node {\midarrow} ++(1.5,0,0);

%% Dense 1: 49920 -> 64
\pic[shift={(7,0,0)}] at (pool3-east) {Box={
    name=dense1,
    caption=,
    fill=\DenseColor,
    height=12,
    width=1.5,
    depth=12}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LATENT SPACE (Bottleneck)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(5,0,0)}] at (dense1-east) {Box={
    name=latent,
    caption=,
    fill=\LatentColor,
    height=8,
    width=2,
    depth=8}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DECODER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Dense 2: 32 -> 64
\pic[shift={(5,0,0)}] at (latent-east) {Box={
    name=dense2,
    caption=,
    fill=\DenseColor,
    height=12,
    width=1.5,
    depth=12}};

%% Dense 3: 64 -> 49920
\pic[shift={(4,0,0)}] at (dense2-east) {Box={
    name=dense3,
    caption=,
    fill=\DenseColor,
    height=22,
    width=1.5,
    depth=15}};

%% Reshape indicator
\draw[connection] (dense3-east) -- node {\midarrow} ++(1.5,0,0);

%% Deconv Block 1: 256 -> 128
\pic[shift={(7,0,0)}] at (dense3-east) {RightBandedBox={
    name=deconv1,
    caption=,
    fill=\ConvColor,
    bandfill=\ConvReluColor,
    height=32,
    width=12,
    depth=25}};

%% Deconv Block 2: 128 -> 64
\pic[shift={(5,0,0)}] at (deconv1-east) {RightBandedBox={
    name=deconv2,
    caption=,
    fill=\ConvColor,
    bandfill=\ConvReluColor,
    height=42,
    width=8,
    depth=35}};

%% Output layer
\pic[shift={(5,0,0)}] at (deconv2-east) {RightBandedBox={
    name=output,
    caption=,
    fill=\ConvColor,
    bandfill=\SoftmaxColor,
    height=50,
    width=2,
    depth=45}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Connections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\draw[connection] (input-east) -- node {\midarrow} (conv1-west);
\draw[connection] (conv1-east) -- node {\midarrow} (pool1-west);
\draw[connection] (pool1-east) -- node {\midarrow} (conv2-west);
\draw[connection] (conv2-east) -- node {\midarrow} (pool2-west);
\draw[connection] (pool2-east) -- node {\midarrow} (conv3-west);
\draw[connection] (conv3-east) -- node {\midarrow} (pool3-west);
\draw[connection] (dense1-east) -- node {\midarrow} (latent-west);
\draw[connection] (latent-east) -- node {\midarrow} (dense2-west);
\draw[connection] (dense2-east) -- node {\midarrow} (dense3-west);
\draw[connection] (deconv1-east) -- node {\midarrow} (deconv2-west);
\draw[connection] (deconv2-east) -- node {\midarrow} (output-west);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Labels - Layer names and dimensions below each layer
%% Using layer-south anchor for proper alignment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ENCODER LABELS
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]input-south) {\textbf{""" + LAYER_LABELS["input"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["input"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]conv1-south) {\textbf{""" + LAYER_LABELS["conv1"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["conv1"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]pool1-south) {\textbf{""" + LAYER_LABELS["pool1"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["pool1"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]conv2-south) {\textbf{""" + LAYER_LABELS["conv2"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["conv2"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]pool2-south) {\textbf{""" + LAYER_LABELS["pool2"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["pool2"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]conv3-south) {\textbf{""" + LAYER_LABELS["conv3"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["conv3"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]pool3-south) {\textbf{""" + LAYER_LABELS["pool3"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["pool3"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]dense1-south) {\textbf{""" + LAYER_LABELS["dense1"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["dense1"]) + r"""}};

%% BOTTLENECK LABEL
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]latent-south) {\textbf{""" + LAYER_LABELS["latent"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["latent"]) + r"""}};

%% DECODER LABELS
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]dense2-south) {\textbf{""" + LAYER_LABELS["dense2"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["dense2"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]dense3-south) {\textbf{""" + LAYER_LABELS["dense3"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["dense3"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]deconv1-south) {\textbf{""" + LAYER_LABELS["deconv1"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["deconv1"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]deconv2-south) {\textbf{""" + LAYER_LABELS["deconv2"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["deconv2"]) + r"""}};
\node[align=center, anchor=north, font=\LARGE] at ([yshift=-3.5cm]output-south) {\textbf{""" + LAYER_LABELS["output"] + r"""}\\{\LARGE """ + format_dims(LAYER_DIMS["output"]) + r"""}};


\end{tikzpicture}
\end{document}
"""

# Save the LaTeX file
output_dir = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/Pytorch_scripts"
output_file = os.path.join(output_dir, "autoencoder_architecture.tex")

with open(output_file, 'w') as f:
    f.write(latex_code)

print("="*70)
print("PlotNeuralNet LaTeX Code Generated!")
print("="*70)
print(f"\nSaved to: {output_file}")
print("\nTO GENERATE THE PDF:")
print("-" * 70)
print("1. Clone PlotNeuralNet:")
print("   cd /Users/oceaneboulais/Github/ThodeLab")
print("   git clone https://github.com/HarisIqbal88/PlotNeuralNet.git")
print()
print("2. Copy the .tex file:")
print("   cp BowheadDeepLearningMATLAB/Pytorch_scripts/autoencoder_architecture.tex \\")
print("      PlotNeuralNet/")
print()
print("3. Compile:")
print("   cd PlotNeuralNet")
print("   pdflatex autoencoder_architecture.tex")
print()
print("4. View:")
print("   open autoencoder_architecture.pdf")
print()
print("="*70)
print("\nARCHITECTURE SUMMARY:")
print("-" * 70)
print("INPUT: 121×104×1 spectrogram")
print("  ↓ Conv2D(64) + BatchNorm + ReLU + MaxPool(2)")
print("  → 60×52×64")
print("  ↓ Conv2D(128) + BatchNorm + ReLU + MaxPool(2)")
print("  → 30×26×128")
print("  ↓ Conv2D(256) + BatchNorm + ReLU + MaxPool(2)")
print("  → 15×13×256")
print("  ↓ Flatten → Linear(49920→64) → Linear(64→32)")
print("  → 32-D LATENT SPACE (bottleneck)")
print("  ↓ Linear(32→64) → Linear(64→49920)")
print("  → 15×13×256")
print("  ↓ ConvTranspose2D(128) + BatchNorm + ReLU")
print("  → 30×26×128")
print("  ↓ ConvTranspose2D(64) + BatchNorm + ReLU")
print("  → 60×52×64")
print("  ↓ ConvTranspose2D(1)")
print("OUTPUT: 121×104×1 reconstructed spectrogram")
print("="*70)
