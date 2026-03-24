#!/bin/bash
# Installation script for WSSS 2026 Research Project
# Run with: bash install.sh

set -e  # Exit on error

echo "================================================"
echo "WSSS 2026 Research - Installation Script"
echo "================================================"
echo ""

# Check Python version
echo "Step 1: Checking Python version..."
python_version=$(python3 --version | cut -d' ' -f2)
python_major=$(echo $python_version | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d'.' -f2)
echo "Python version: $python_version"

# Check if Python >= 3.9
if [ "$python_major" -lt 3 ] || { [ "$python_major" -eq 3 ] && [ "$python_minor" -lt 9 ]; }; then
    echo "ERROR: Python 3.9+ is required (found $python_version)"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: Not in a virtual environment!"
    echo "Recommended: source venv/bin/activate"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Upgrade pip
echo "Step 2: Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Step 3: Install core dependencies
echo "Step 3: Installing core dependencies (PyTorch, NumPy, etc.)..."
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy>=1.24.0 opencv-python>=4.7.0 pillow>=9.5.0 tqdm>=4.65.0
pip install protobuf==3.20.3
echo "✓ Core dependencies installed"
echo ""

# Step 4: Install data augmentation
echo "Step 4: Installing data augmentation libraries..."
pip install albumentations>=1.3.0
echo "✓ Albumentations installed"
echo ""

# Step 5: Install model libraries
echo "Step 5: Installing model architectures (timm, transformers)..."
pip install timm>=0.9.0
pip install transformers==4.35.2
echo "✓ Model libraries installed"
echo ""

# Step 6: Install medical imaging models
echo "Step 6: Installing medical imaging models..."
pip install medclip==0.0.3
echo "✓ MedCLIP installed"
echo ""

echo "Step 6b: Installing CONCH (optional, may take 5-10 min)..."
read -p "Install CONCH? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    pip install git+https://github.com/Mahmoodlab/CONCH.git
    echo "✓ CONCH installed"
else
    echo "⊘ CONCH skipped (can install later with: pip install git+https://github.com/Mahmoodlab/CONCH.git)"
fi
echo ""

# Step 7: Install configuration management
echo "Step 7: Installing configuration management..."
pip install omegaconf>=2.3.0 pyyaml>=6.0
echo "✓ Config libraries installed"
echo ""

# Step 8: Install utilities
echo "Step 8: Installing utilities..."
pip install scikit-learn>=1.2.0 matplotlib>=3.7.0 seaborn>=0.12.0 icecream>=2.1.0
pip install ttach>=0.0.3
echo "✓ Utilities installed"
echo ""

# Step 9: Install fast_slic
echo "Step 9: Installing fast_slic..."
pip install fast_slic>=0.4.0 || echo "⚠ fast_slic failed (optional, can skip)"
echo ""

# Step 10: Install pydensecrf (optional, requires build tools)
echo "Step 10: Installing pydensecrf for CRF post-processing (optional)..."
read -p "Install pydensecrf? Requires build-essential python3-dev (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    # Check if build tools are available
    if ! command -v gcc &> /dev/null; then
        echo "⚠ WARNING: gcc not found. Install build tools first:"
        echo "  sudo apt-get install build-essential python3-dev"
        echo "⊘ pydensecrf skipped"
    else
        pip install git+https://github.com/lucasb-eyer/pydensecrf.git || {
            echo "⚠ pydensecrf installation failed (optional, can skip)"
        }
        echo "✓ pydensecrf installed"
    fi
else
    echo "⊘ pydensecrf skipped"
fi
echo ""

# Step 11: Install HuggingFace Hub
echo "Step 11: Installing HuggingFace Hub..."
pip install huggingface-hub>=0.15.0
echo "✓ HuggingFace Hub installed"
echo ""

# Step 12: Verify installation
echo "================================================"
echo "Step 12: Verifying installation..."
echo "================================================"
echo ""

echo -n "PyTorch: "
python3 -c "import torch; print(f'v{torch.__version__} (CUDA: {torch.cuda.is_available()})')" || echo "❌ FAILED"

echo -n "NumPy: "
python3 -c "import numpy; print(f'v{numpy.__version__}')" || echo "❌ FAILED"

echo -n "OpenCV: "
python3 -c "import cv2; print(f'v{cv2.__version__}')" || echo "❌ FAILED"

echo -n "timm: "
python3 -c "import timm; print(f'v{timm.__version__}')" || echo "❌ FAILED"

echo -n "transformers: "
python3 -c "import transformers; print(f'v{transformers.__version__}')" || echo "❌ FAILED"

echo -n "MedCLIP: "
python3 -c "import medclip; print('v0.0.3')" || echo "❌ FAILED"

echo -n "CONCH: "
python3 -c "from conch.open_clip_custom import create_model_from_pretrained; print('✓')" 2>/dev/null || echo "⊘ not installed (optional)"

echo -n "OmegaConf: "
python3 -c "import omegaconf; print(f'v{omegaconf.__version__}')" || echo "❌ FAILED"

echo -n "albumentations: "
python3 -c "import albumentations; print(f'v{albumentations.__version__}')" || echo "❌ FAILED"

echo -n "pydensecrf: "
python3 -c "import pydensecrf; print('✓')" 2>/dev/null || echo "⊘ not installed (optional)"

echo ""
echo "Verifying project imports..."
echo -n "src.models.network: "
python3 -c "from src.models.network import ClsNetwork; print('✓')" || echo "❌ FAILED"

echo -n "src.datasets: "
python3 -c "from src.datasets.bcss_dataset import get_cls_dataset; print('✓')" || echo "❌ FAILED"

echo -n "src.losses: "
python3 -c "from src.losses.contrast_loss import InfoNCELossFG; print('✓')" || echo "❌ FAILED"

echo -n "src.utils: "
python3 -c "from src.utils.evaluate import validate; print('✓')" || echo "❌ FAILED"

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Configure your data paths in configs/default.yaml"
echo "  2. Set your HuggingFace token in configs/default.yaml"
echo "  3. Run training: python scripts/train.py --config configs/default.yaml --gpu 0"
echo ""
echo "See QUICKSTART.md for detailed usage instructions."
