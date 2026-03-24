# Project Restructuring Guide

This document explains how the monolithic `wsss.py` file was restructured into a modular, maintainable project.

## Original Structure

```
wsss.py (2700+ lines)
├── All imports
├── Dataset classes
├── Utility functions
├── Loss functions
├── Model components
├── Training loop
└── Main execution
```

## New Modular Structure

### 1. **Source Code Organization** (`src/`)

#### `src/datasets/`
Responsible for data loading and preprocessing.

**Files:**
- `bcss_dataset.py`: Contains BCSSTrainingDataset, BCSSTestDataset, BCSSWSSSDataset
- `__init__.py`: Exports dataset classes

**From wsss.py lines:** 58-176

#### `src/losses/`
All loss functions isolated for easy modification and testing.

**Files:**
- `contrast_loss.py`: InfoNCELossFG, InfoNCELossBG, MultiResConsistencyLoss
- `__init__.py`: Exports loss classes

**From wsss.py lines:** 395-469

#### `src/utils/`
Reusable utility functions grouped by functionality.

**Files:**
- `pyutils.py`: General utilities (set_seed, AverageMeter, etc.)
- `optimizer.py`: PolyWarmupAdamW custom optimizer
- `evaluate.py`: Evaluation metrics and ConfusionMatrix
- `hierarchical.py`: Hierarchical classification helpers
- `fgbg_feature.py`: Foreground/background feature extraction
- `__init__.py`: Central exports

**From wsss.py lines:** 179-243, 248-280, 285-393, 472-606, 609-739

#### `src/models/`
Model architecture components (to be created from remaining code).

**Structure:**
```
src/models/
├── __init__.py
├── encoder.py          # UNI encoder + LoRA
├── fpn.py              # SimpleFeaturePyramid
├── vq_modules.py       # VQCodebook, VQCAMHead
├── hash_miner.py       # HashHardNegativeMiner
├── text_encoder.py     # Text prototype encoding
└── network.py          # Main ClsNetwork class
```

**Content mapping:**
- `encoder.py`: UNIEncoder, LoRALinear (lines 1458-1540)
- `fpn.py`: SimpleFeaturePyramid (lines 1543-1585)
- `vq_modules.py`: VQCodebook, VQCAMHead (lines 742-903)
- `hash_miner.py`: HashHardNegativeMiner, HashMemoryBank (lines 923-1041)
- `text_encoder.py`: _encode_text_prompts, TISSUE_TEXT_PROMPTS (lines 1601-1697)
- `network.py`: ClsNetwork, AdaptiveLayer (lines 1588-1960)

### 2. **Configuration Management** (`configs/`)

Instead of inline YAML, configs are now external files:
- `default.yaml`: All training parameters
- `inference.yaml`: Inference-specific settings

**Benefits:**
- Easy experimentation (just change config file)
- No code modification needed for hyperparameter tuning
- Version control friendly

**From wsss.py lines:** 1963-2061

### 3. **Executable Scripts** (`scripts/`)

Main entry points for different tasks:

#### `scripts/train.py`
Full training pipeline:
- Argument parsing
- Config loading
- Dataset preparation
- Model initialization
- Training loop with validation
- Checkpoint saving

**From wsss.py lines:** 2303-2700 (train function + main)

#### `scripts/inference.py`
Generate predictions:
- Load trained model
- Process test images
- Generate CAMs
- Save segmentation masks

**From wsss.py lines:** 1272-1324 (generate_cam function)

#### `scripts/visualize.py`
Framework visualization:
- Export multi-scale CAMs
- Create beautiful figures for papers
- VQ codebook analysis

**From wsss.py lines:** 2064-2300 (export_framework_images)

### 4. **Documentation** (`docs/`)

Comprehensive project documentation:

#### `ARCHITECTURE.md`
Detailed explanation of:
- Model architecture
- Information flow
- Design decisions
- Ablation studies

#### `TRAINING.md`
Step-by-step training guide:
- Data preparation
- Environment setup
- Training commands
- Hyperparameter tuning
- Troubleshooting

#### `API.md`
API reference for:
- Dataset classes
- Model classes
- Utility functions
- Configuration schema

## Migration Steps

### Step 1: Split Dataset Code

```python
# wsss.py (old)
class BCSSTrainingDataset(Dataset):
    # ... 30 lines ...

# src/datasets/bcss_dataset.py (new)
class BCSSTrainingDataset(Dataset):
    # ... same code, properly documented ...
```

### Step 2: Extract Utilities

```python
# wsss.py (old)
def set_seed(seed): ...
class AverageMeter: ...

# src/utils/pyutils.py (new)
def set_seed(seed):
    """Set random seed for reproducibility"""
    # ... implementation ...
```

### Step 3: Isolate Model Components

```python
# wsss.py (old)
class ClsNetwork(nn.Module):
    def __init__(self, ...):
        # 200 lines of initialization
    def forward(self, x):
        # 100 lines of forward pass

# src/models/ (new)
# encoder.py: UNI + LoRA
# fpn.py: Feature pyramid
# vq_modules.py: VQ codebook
# network.py: Main network (uses above components)
```

### Step 4: Create Training Script

```python
# wsss.py (old)
def train(cfg):
    # 400 lines of training code

if __name__ == "__main__":
    cfg = OmegaConf.create(...)
    train(cfg)

# scripts/train.py (new)
import argparse
from omegaconf import OmegaConf
from src.models import ClsNetwork
from src.datasets import BCSSTrainingDataset
# ... clean imports ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    # ... clean argument parsing ...
```

## Benefits of New Structure

### 1. **Maintainability**
- Each file has a single, clear purpose
- Easy to locate and fix bugs
- Changes don't ripple through entire codebase

### 2. **Testability**
- Can unit test individual components
- Mock dependencies easily
- Faster development cycle

### 3. **Reusability**
- Import only what you need
- Use components in other projects
- Clear API boundaries

### 4. **Collaboration**
- Multiple people can work on different files
- Fewer merge conflicts
- Clear ownership of code sections

### 5. **Extensibility**
- Add new datasets: just add to `src/datasets/`
- New loss function: add to `src/losses/`
- Custom model: implement in `src/models/`

## Example: Adding a New Dataset

### Old Way (monolithic)
```python
# Edit wsss.py at line 180
class MyNewDataset(Dataset):
    # ... implement new dataset among 2700 lines ...
```

### New Way (modular)
```python
# Create src/datasets/my_dataset.py
class MyNewDataset(Dataset):
    """My new dataset for X task"""
    # ... clean implementation ...

# Update src/datasets/__init__.py
from .my_dataset import MyNewDataset
__all__ = [..., 'MyNewDataset']

# Use in scripts/train.py
from src.datasets import MyNewDataset
```

## Next Steps

To complete the restructuring:

1. ✅ Create folder structure
2. ✅ Extract datasets → `src/datasets/`
3. ✅ Extract losses → `src/losses/`
4. ✅ Extract utils → `src/utils/`
5. 🔄 Extract model components → `src/models/`
   - Create `encoder.py`
   - Create `fpn.py`
   - Create `vq_modules.py`
   - Create `hash_miner.py`
   - Create `network.py`
6. 🔄 Create training script → `scripts/train.py`
7. 🔄 Create inference script → `scripts/inference.py`
8. 🔄 Create visualization script → `scripts/visualize.py`
9. ✅ Write documentation → `README.md`, `docs/`

## Code Quality Improvements

Along with restructuring, we've improved:

1. **Type Hints**: Add type annotations for better IDE support
2. **Docstrings**: Clear documentation for all public functions
3. **Error Handling**: Proper exception handling with informative messages
4. **Logging**: Structured logging instead of print statements
5. **Configuration**: External configs instead of hardcoded values
6. **Testing**: Unit tests for critical components

## Example Transformation

### Before (monolithic wsss.py)
```python
# Line 1500+
def train(cfg):
    device = torch.device(f"cuda:{GPU_ID}")
    set_seed(42)
    clip_model = MedCLIPModel(...)
    # ... 400 more lines ...
```

### After (modular scripts/train.py)
```python
from src.models import ClsNetwork
from src.datasets import BCSSTrainingDataset
from src.utils import set_seed, PolyWarmupAdamW

def setup_training(cfg):
    """Setup training environment and components"""
    device = torch.device(f"cuda:{cfg.gpu_id}")
    set_seed(cfg.seed)
    return device

def train(cfg):
    """Main training loop"""
    device = setup_training(cfg)
    model = ClsNetwork(**cfg.model)
    # ... clean, organized code ...
```

---

This modular structure makes the codebase professional, maintainable, and ready for publication on GitHub!
