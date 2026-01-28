#!/bin/bash
# Script to create new clean 'deem' repository

set -e

echo "================================================"
echo "Creating New Clean DEEM Repository"
echo "================================================"

# 1. Create fresh directory for new repo
echo ""
echo "Step 1: Creating fresh directory..."
cd /home/dsi/maymona3
rm -rf deem 2>/dev/null || true
mkdir deem
cd deem
git init
git branch -m main  # Set main as default branch

# 2. Copy clean package files from refactoring branch
echo ""
echo "Step 2: Copying package files from refactoring branch..."
cd /home/dsi/maymona3/rbm_python
git checkout refactoring

# Copy production files
rsync -av --progress \
    --exclude='.git' \
    --exclude='src/' \
    --exclude='datasets/' \
    --exclude='archive/' \
    --exclude='analysis/' \
    --exclude='best_*.txt' \
    --exclude='best_*.pth' \
    --exclude='optuna_*.txt' \
    --exclude='*.sqlite3' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='dist/' \
    --exclude='build/' \
    --exclude='*.egg-info' \
    ./ /home/dsi/maymona3/deem/

# 3. Commit to new repo
echo ""
echo "Step 3: Creating initial commit..."
cd /home/dsi/maymona3/deem
git add -A
git commit -m "Initial commit: DEEM v0.2.0 production package

- Clean scikit-learn compatible API
- Bundled AutoML models (works out-of-box)
- Comprehensive documentation
- Full test suite
- Ready for PyPI publication"

# 4. Update repository URLs in files
echo ""
echo "Step 4: Updating repository URLs..."
sed -i 's|github.com/Rem4rkable/rbm_python|github.com/Rem4rkable/deem|g' pyproject.toml
sed -i 's|github.com/Rem4rkable/rbm_python|github.com/Rem4rkable/deem|g' README.md
git add pyproject.toml README.md
git commit -m "Update repository URLs to new deem repo"

# 5. Create experiments branch from rebuttal
echo ""
echo "Step 5: Creating experiments branch from rebuttal..."
cd /home/dsi/maymona3/rbm_python
git checkout rebuttal

# Copy research files to temp
rm -rf /tmp/deem-experiments 2>/dev/null || true
mkdir -p /tmp/deem-experiments
rsync -av --progress \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    ./ /tmp/deem-experiments/

# Create experiments branch in new repo
cd /home/dsi/maymona3/deem
git checkout -b experiments

# Copy research files
rsync -av /tmp/deem-experiments/ ./
git add -A
git commit -m "Add experiments branch for paper reproduction

Research code for reproducing paper results:
- src/ directory with all research scripts
- datasets/ with experiment data
- analysis/ with experiment results
- Optuna hyperparameter search (accuracy removed)

This branch preserves the complete research environment."

# Switch back to main
git checkout main

echo ""
echo "================================================"
echo "✅ New Repository Created Successfully!"
echo "================================================"
echo ""
echo "Location: /home/dsi/maymona3/deem"
echo ""
echo "Branches:"
echo "  - main: Clean production package"
echo "  - experiments: Research code from rebuttal"
echo ""
echo "Next Steps:"
echo "1. Create new GitHub repo: https://github.com/new"
echo "   Name: deem"
echo "   Visibility: Public"
echo ""
echo "2. Push to GitHub:"
echo "   cd /home/dsi/maymona3/deem"
echo "   git remote add origin https://github.com/Rem4rkable/deem.git"
echo "   git push -u origin main"
echo "   git push -u origin experiments"
echo ""
echo "3. Continue working in: /home/dsi/maymona3/deem"
echo "   This is now your main development directory!"
echo ""
echo "4. Keep /home/dsi/maymona3/rbm_python as archive/backup"
echo ""
