#!/usr/bin/env python3
"""
Simple matplotlib-based architecture diagram for the autoencoder
(Alternative to PlotNeuralNet which requires LaTeX setup)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(10, 9.5, 'Whale Call Autoencoder Architecture', 
        fontsize=24, fontweight='bold', ha='center')

# Color scheme
encoder_color = '#FFD700'  # Gold
latent_color = '#FF1493'   # Deep pink
decoder_color = '#87CEEB'  # Sky blue
arrow_color = '#333333'

y_center = 5

# ENCODER
x = 1
# Input
ax.add_patch(FancyBboxPatch((x, y_center-0.5), 0.8, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='white', edgecolor='black', linewidth=2))
ax.text(x+0.4, y_center, 'Input\n121×104×1', ha='center', va='center', fontsize=9)

x += 1.2
# Conv1
ax.add_patch(FancyBboxPatch((x, y_center-0.8), 1, 1.6,
                             boxstyle="round,pad=0.1",
                             facecolor=encoder_color, edgecolor='black', linewidth=2))
ax.text(x+0.5, y_center+0.5, 'Conv1', ha='center', fontweight='bold', fontsize=10)
ax.text(x+0.5, y_center, '64 filters', ha='center', fontsize=8)
ax.text(x+0.5, y_center-0.5, '60×52×64', ha='center', fontsize=8)

x += 1.5
# Conv2
ax.add_patch(FancyBboxPatch((x, y_center-1), 1.2, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=encoder_color, edgecolor='black', linewidth=2))
ax.text(x+0.6, y_center+0.6, 'Conv2', ha='center', fontweight='bold', fontsize=10)
ax.text(x+0.6, y_center, '128 filters', ha='center', fontsize=8)
ax.text(x+0.6, y_center-0.6, '30×26×128', ha='center', fontsize=8)

x += 1.8
# Conv3
ax.add_patch(FancyBboxPatch((x, y_center-1.2), 1.4, 2.4,
                             boxstyle="round,pad=0.1",
                             facecolor=encoder_color, edgecolor='black', linewidth=2))
ax.text(x+0.7, y_center+0.7, 'Conv3', ha='center', fontweight='bold', fontsize=10)
ax.text(x+0.7, y_center, '256 filters', ha='center', fontsize=8)
ax.text(x+0.7, y_center-0.7, '15×13×256', ha='center', fontsize=8)

x += 2
# Flatten + Dense
ax.add_patch(FancyBboxPatch((x, y_center-0.6), 1, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=encoder_color, edgecolor='black', linewidth=2))
ax.text(x+0.5, y_center+0.3, 'Flatten', ha='center', fontweight='bold', fontsize=9)
ax.text(x+0.5, y_center-0.3, 'Dense 64', ha='center', fontsize=8)

# LATENT SPACE
x += 1.5
ax.add_patch(FancyBboxPatch((x, y_center-1.5), 1.5, 3,
                             boxstyle="round,pad=0.15",
                             facecolor=latent_color, edgecolor='black', linewidth=3))
ax.text(x+0.75, y_center+0.8, 'LATENT', ha='center', fontweight='bold', fontsize=14)
ax.text(x+0.75, y_center+0.3, 'SPACE', ha='center', fontweight='bold', fontsize=14)
ax.text(x+0.75, y_center-0.3, '32-D', ha='center', fontsize=12, fontweight='bold')
ax.text(x+0.75, y_center-0.8, 'Bottleneck', ha='center', fontsize=9, style='italic')

# DECODER
x += 2
# Dense
ax.add_patch(FancyBboxPatch((x, y_center-0.6), 1, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=decoder_color, edgecolor='black', linewidth=2))
ax.text(x+0.5, y_center+0.3, 'Dense', ha='center', fontweight='bold', fontsize=9)
ax.text(x+0.5, y_center-0.3, 'Reshape', ha='center', fontsize=8)

x += 1.5
# Deconv1
ax.add_patch(FancyBboxPatch((x, y_center-1.2), 1.4, 2.4,
                             boxstyle="round,pad=0.1",
                             facecolor=decoder_color, edgecolor='black', linewidth=2))
ax.text(x+0.7, y_center+0.7, 'Deconv1', ha='center', fontweight='bold', fontsize=10)
ax.text(x+0.7, y_center, '128 filters', ha='center', fontsize=8)
ax.text(x+0.7, y_center-0.7, '30×26×128', ha='center', fontsize=8)

x += 1.8
# Deconv2
ax.add_patch(FancyBboxPatch((x, y_center-1), 1.2, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=decoder_color, edgecolor='black', linewidth=2))
ax.text(x+0.6, y_center+0.6, 'Deconv2', ha='center', fontweight='bold', fontsize=10)
ax.text(x+0.6, y_center, '64 filters', ha='center', fontsize=8)
ax.text(x+0.6, y_center-0.6, '60×52×64', ha='center', fontsize=8)

x += 1.5
# Deconv3
ax.add_patch(FancyBboxPatch((x, y_center-0.8), 1, 1.6,
                             boxstyle="round,pad=0.1",
                             facecolor=decoder_color, edgecolor='black', linewidth=2))
ax.text(x+0.5, y_center+0.5, 'Deconv3', ha='center', fontweight='bold', fontsize=10)
ax.text(x+0.5, y_center, '1 filter', ha='center', fontsize=8)
ax.text(x+0.5, y_center-0.5, '121×104×1', ha='center', fontsize=8)

x += 1.3
# Output
ax.add_patch(FancyBboxPatch((x, y_center-0.5), 0.8, 1,
                             boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor='black', linewidth=2))
ax.text(x+0.4, y_center, 'Output\n121×104×1', ha='center', va='center', fontsize=9)

# Add arrows between blocks
arrow_props = dict(arrowstyle='->', lw=2, color=arrow_color)
arrow_y = y_center

positions = [1.8, 3.5, 5.5, 7.6, 9.2, 11.5, 13.2, 15.2, 17, 18.5]
for i in range(len(positions)-1):
    arrow = FancyArrowPatch((positions[i], arrow_y), (positions[i+1], arrow_y),
                            **arrow_props)
    ax.add_patch(arrow)

# Labels
ax.text(4, 8.5, 'ENCODER', fontsize=16, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor=encoder_color, alpha=0.7))
ax.text(10.25, 8.5, 'LATENT', fontsize=16, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor=latent_color, alpha=0.7))
ax.text(15.5, 8.5, 'DECODER', fontsize=16, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor=decoder_color, alpha=0.7))

# Architecture details box
details_text = """Architecture Details:
• Total Parameters: 6,979,169
• Compression: 377:1 (12,584 → 32)
• Loss: MSE (Mean Squared Error)
• Optimizer: Adam (lr=0.001)
• Batch Normalization after each conv
• ReLU activation throughout
• MaxPool(2) after each encoder conv"""

ax.text(10, 1.5, details_text, fontsize=10, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.8, linewidth=2))

# Data flow annotation
ax.annotate('', xy=(10.25, 3.5), xytext=(10.25, 6.5),
            arrowprops=dict(arrowstyle='<->', lw=3, color='red'))
ax.text(10.8, 5, 'Compress\n377×', fontsize=10, fontweight='bold', color='red')

plt.tight_layout()

# Save
output_path = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/Pytorch_scripts/autoencoder_architecture.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved architecture diagram to: {output_path}")

plt.show()
