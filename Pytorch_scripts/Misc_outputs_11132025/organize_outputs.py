#!/usr/bin/env python3
"""
Organize training artifacts into best-practice folders.

Moves/copies:
- *.pth  -> <repo_root>/models[/<dataset_slug>]/
- *.png  -> <repo_root>/plots[/<dataset_slug>]/
- *.m    -> <repo_root>/matlab/

Dataset slug extraction is based on common filename patterns used in this repo, e.g.:
  improved_<dataset>_final_model.pth
  improved_<dataset>_best_model.pth
  improved_<dataset>_training_plot.png
  improved_<dataset>_final_comparison.png
  improved_<dataset>_random15_panel.png

If a filename does not match known patterns, it's placed directly under models/ or plots/.

Usage examples:
  python organize_outputs.py                 # move files into models/ and plots/
  python organize_outputs.py --dry-run       # preview moves without changing files
    python organize_outputs.py --copy          # copy instead of move
    python organize_outputs.py --models-dir ./artifacts/models --plots-dir ./artifacts/plots --matlab-dir ./artifacts/matlab
"""
import argparse
import os
import re
import shutil
from typing import Optional, Tuple


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MODEL_SUFFIXES = [
    "_final_model.pth",
    "_best_model.pth",
    ".dir_model.pth",
]

PLOT_SUFFIXES = [
    "_training_plot.png",
    "_final_comparison.png",
    "_random15_panel.png",
]


def extract_dataset_slug(filename: str) -> Optional[str]:
    """Try to extract dataset slug from common 'improved_<dataset>_<suffix>' patterns.
    Returns None if no pattern is recognized.
    """
    base = os.path.basename(filename)
    if base.startswith("improved_"):
        stem = base[len("improved_"):]
        # Try model suffixes first
        for suf in MODEL_SUFFIXES + PLOT_SUFFIXES:
            if stem.endswith(suf):
                return stem[: -len(suf)] if len(suf) > 0 else stem
        # Fallback: if it still has an extension but no known suffix, strip last token after last underscore
        # This is conservative: put those into root if ambiguous
        return None
    return None


def should_skip_dir(dirpath: str) -> bool:
    parts = set(dirpath.split(os.sep))
    skip_names = {
        ".git", "__pycache__", "runs", "models", "plots", "matlab",
        ".venv", "venv", "env", ".idea", ".vscode"
    }
    return any(name in parts for name in skip_names)


def plan_moves(root: str) -> Tuple[list, list, list]:
    """Scan repository and return planned moves for (models, plots, matlab).
    Each models/plots list contains tuples: (src_path, dataset_slug_or_None)
    Matlab list contains src paths.
    """
    model_moves = []
    plot_moves = []
    matlab_moves = []
    for dirpath, dirnames, filenames in os.walk(root):
        if should_skip_dir(dirpath):
            continue
        for fname in filenames:
            src = os.path.join(dirpath, fname)
            # Skip hidden files
            if fname.startswith('.'):
                continue
            if fname.lower().endswith('.pth'):
                slug = extract_dataset_slug(fname)
                model_moves.append((src, slug))
            elif fname.lower().endswith('.png'):
                slug = extract_dataset_slug(fname)
                plot_moves.append((src, slug))
            elif fname.lower().endswith('.m'):
                matlab_moves.append(src)
    return model_moves, plot_moves, matlab_moves


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Organize .pth, .png, and .m artifacts into models/, plots/, and matlab/ folders")
    parser.add_argument("--repo-root", default=REPO_ROOT, help="Repository root to scan")
    parser.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models"), help="Destination folder for .pth files")
    parser.add_argument("--plots-dir", default=os.path.join(REPO_ROOT, "plots"), help="Destination folder for .png files")
    parser.add_argument("--matlab-dir", default=os.path.join(REPO_ROOT, "matlab"), help="Destination folder for .m files")
    parser.add_argument("--copy", action='store_true', help="Copy files instead of moving them")
    parser.add_argument("--dry-run", action='store_true', help="Only print what would happen")
    args = parser.parse_args()

    model_moves, plot_moves, matlab_moves = plan_moves(args.repo_root)
    if not model_moves and not plot_moves and not matlab_moves:
        print("No .pth, .png, or .m files found to organize.")
        return

    op_name = "COPY" if args.copy else "MOVE"

    # Execute model moves
    for src, slug in model_moves:
        dest_dir = os.path.join(args.models_dir, slug) if slug else args.models_dir
        ensure_dir(dest_dir)
        dest = os.path.join(dest_dir, os.path.basename(src))
        if args.dry_run:
            print(f"[DRY-RUN] {op_name} model: {src} -> {dest}")
        else:
            print(f"{op_name} model: {src} -> {dest}")
            if args.copy:
                shutil.copy2(src, dest)
            else:
                shutil.move(src, dest)

    # Execute plot moves
    for src, slug in plot_moves:
        dest_dir = os.path.join(args.plots_dir, slug) if slug else args.plots_dir
        ensure_dir(dest_dir)
        dest = os.path.join(dest_dir, os.path.basename(src))
        if args.dry_run:
            print(f"[DRY-RUN] {op_name} plot: {src} -> {dest}")
        else:
            print(f"{op_name} plot: {src} -> {dest}")
            if args.copy:
                shutil.copy2(src, dest)
            else:
                shutil.move(src, dest)

    # Execute matlab moves
    for src in matlab_moves:
        dest_dir = args.matlab_dir
        ensure_dir(dest_dir)
        dest = os.path.join(dest_dir, os.path.basename(src))
        if args.dry_run:
            print(f"[DRY-RUN] {op_name} matlab: {src} -> {dest}")
        else:
            print(f"{op_name} matlab: {src} -> {dest}")
            if args.copy:
                shutil.copy2(src, dest)
            else:
                shutil.move(src, dest)

    print("Done organizing artifacts.")


if __name__ == "__main__":
    main()
