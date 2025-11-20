# Git LFS Quota Fix - Step-by-Step Guide

## Problem
The Git LFS quota has been exceeded due to large model files (.pth), data files (.mat), images, and result directories in the repository history.

## Solution Overview
1. Update .gitignore to prevent future commits of large files ✓ (DONE)
2. Remove large files from Git history
3. Force push the cleaned history

---

## Quick Fix (Manual Method)

### Step 1: Remove large files from current tracking
```bash
cd /Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB

# Create backup branch first
git branch backup-before-cleanup

# Remove currently tracked large files
git rm -r --cached --ignore-unmatch "Pytorch_scripts/runs"
git rm -r --cached --ignore-unmatch "Pytorch_scripts/plots"
git rm -r --cached --ignore-unmatch "Pytorch_scripts/models"
git rm -r --cached --ignore-unmatch "Pytorch_scripts/results"
git rm -r --cached --ignore-unmatch "Pytorch_scripts/Misc_outputs*"
git rm -r --cached --ignore-unmatch "Pytorch_scripts/Autoencoder_*"
git rm -r --cached --ignore-unmatch "Pytorch_scripts/trained_models"
git rm -r --cached --ignore-unmatch "plots"
git rm -r --cached --ignore-unmatch "models"
git rm -r --cached --ignore-unmatch "results"

# Keep essential MATLAB file
git add -f GSI_header_table.mat

# Commit the changes
git add .gitignore
git commit -m "Remove large files and update .gitignore"
```

### Step 2: Clean Git history using BFG Repo-Cleaner (RECOMMENDED)

Install BFG:
```bash
brew install bfg
```

Clean the repository:
```bash
# Remove files larger than 1MB from history (keeps current HEAD)
bfg --strip-blobs-bigger-than 1M

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Step 3: Force push cleaned history
```bash
git push origin OB-branch --force
```

---

## Alternative: git filter-repo (if BFG doesn't work)

Install git-filter-repo:
```bash
brew install git-filter-repo
```

Remove specific paths:
```bash
git filter-repo --path Pytorch_scripts/runs --invert-paths
git filter-repo --path Pytorch_scripts/plots --invert-paths
git filter-repo --path Pytorch_scripts/models --invert-paths
git filter-repo --path Pytorch_scripts/results --invert-paths
git filter-repo --path plots --invert-paths
git filter-repo --path models --invert-paths
git filter-repo --path results --invert-paths

git push origin OB-branch --force
```

---

## Verification

Check repository size:
```bash
git count-objects -vH
```

Check largest files in history:
```bash
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print $3, $4}' | \
  sort -rn | \
  head -20
```

---

## Preventing Future Issues

### Before committing, ALWAYS check:
```bash
# See what will be committed
git status

# If you see any .pth, .mat, .jpg, .png, or .dir files, DO NOT COMMIT!
# Instead, update .gitignore and remove from tracking:
git rm --cached <large_file>
```

### Safe commit checklist:
- ✓ Only .py, .m, .sh, .md, .txt (small text files) should be committed
- ✗ NEVER commit: .pth, .mat (except GSI_header_table.mat), .jpg, .png, models/, results/, plots/, runs/

### Emergency: If you accidentally committed large files:
```bash
# Immediately undo last commit (keeps changes)
git reset --soft HEAD~1

# Remove large files from staging
git reset HEAD <large_file>

# Add to .gitignore
echo "<large_file>" >> .gitignore

# Commit without large files
git add .
git commit -m "Your commit message"
```

---

## Contact
If cleanup doesn't work or you need help, contact the repository owner (AaronThode) before force pushing.

**WARNING**: Force pushing rewrites history - coordinate with team members!
