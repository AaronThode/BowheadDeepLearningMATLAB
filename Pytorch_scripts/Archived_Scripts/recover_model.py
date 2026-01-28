#!/usr/bin/env python3
"""Emergency model recovery script"""
import torch
import os
from datetime import datetime

# Recreate the output directory
output_dir = "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/Autoencoder_v06_10E_32LD_MostlyManual_Date20251120-193358.dir"
os.makedirs(output_dir, exist_ok=True)

# Check if model is still in memory (unlikely but worth trying)
import sys
frame = sys._getframe()
while frame:
    if 'model' in frame.f_locals:
        model = frame.f_locals['model']
        model_path = os.path.join(output_dir, 'autoencoder_clean.pth')
        torch.save(model.state_dict(), model_path)
        print(f"SUCCESS! Saved model to: {model_path}")
        sys.exit(0)
    frame = frame.f_back

print("Model not found in memory - training session has ended")
print("You'll need to restart the training, but the fixed script will now work correctly")
