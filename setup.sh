#!/bin/bash
# Setup script for nebula-diffusion
# Installs all dependencies including the fast dataloader

set -e

echo "=== Nebula Diffusion Setup ==="

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
    echo "Warning: No virtual environment detected. Consider using venv or conda."
fi

# Install main dependencies
echo ""
echo "Installing main dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tokenizers
pip install wandb tensorboard
pip install pyyaml tqdm
pip install einops

# Install the fast dataloader from GitHub
echo ""
echo "Installing fast dataloader from GitHub..."
pip install git+https://github.com/Orolol/data-loader-fast.git

# Install the nebula package in editable mode (if setup.py exists)
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo ""
    echo "Installing nebula package in editable mode..."
    pip install -e .
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start training:"
echo "  python scripts/train.py --config configs/small.yaml"
echo ""
