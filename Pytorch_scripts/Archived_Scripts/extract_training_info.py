#!/usr/bin/env python3
"""
Extract training information from checkpoint files.
"""
import torch
import sys
import os

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("\n" + "="*70)
    print("TRAINING INFORMATION EXTRACTION")
    print("="*70)
    
    # Check for checkpoint_epoch100.pth
    checkpoint_path = os.path.join(results_dir, "checkpoint_epoch100.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"\nLoading: {os.path.basename(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            print("\nCheckpoint contents:")
            for key in checkpoint.keys():
                if key != 'model_state_dict' and key != 'optimizer_state_dict':
                    print(f"  {key}: {checkpoint[key]}")
            
            if 'epoch' in checkpoint:
                print(f"\n✓ Epochs: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"✓ Final loss: {checkpoint['loss']:.6f}")
        else:
            print("Checkpoint is state dict only, no training metadata")
    else:
        print(f"No checkpoint_epoch100.pth found in {results_dir}")
    
    # Try to infer training time from file timestamps
    print("\n" + "="*70)
    print("ESTIMATING TRAINING TIME FROM FILE TIMESTAMPS")
    print("="*70)
    
    checkpoint_10 = os.path.join(results_dir, "checkpoint_epoch10.pth")
    checkpoint_100 = os.path.join(results_dir, "checkpoint_epoch100.pth")
    
    if os.path.exists(checkpoint_10) and os.path.exists(checkpoint_100):
        time_10 = os.path.getmtime(checkpoint_10)
        time_100 = os.path.getmtime(checkpoint_100)
        
        duration_seconds = time_100 - time_10
        duration_minutes = duration_seconds / 60
        duration_hours = duration_minutes / 60
        
        # This is for 90 epochs (epoch 10 to 100)
        estimated_total_minutes = (duration_minutes / 90) * 100
        estimated_total_hours = estimated_total_minutes / 60
        
        print(f"Time from epoch 10 to epoch 100: {duration_minutes:.1f} min ({duration_hours:.2f} hours)")
        print(f"Estimated total training time: ~{estimated_total_minutes:.1f} min (~{estimated_total_hours:.2f} hours)")
    else:
        print("Cannot estimate - checkpoint files missing")
    
    print("="*70 + "\n")
