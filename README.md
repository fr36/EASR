# EASR: Expression Aware Supervision and Refinement

This is a PyTorch implementation of the EASR framework for facial expression recognition with expression-aware supervision and progressive feature refinement.

## Dataset Setup

The training script expects datasets to be organized in the project root `datasets/` directory with standard ImageFolder structure (train/test splits). Supported datasets include RAF-DB, FERPlus, FER2013, and AffectNet.

## Training

Basic training command:

```bash
python train.py \
--dataset <dataset_name> \
--output-dir <path_to_checkpoints> \
--other parameters for training
```

### Key Parameters

- `--dataset`: Dataset name (default: rafdb)
- `--loss`: Loss function - `ce`, `ce_ls`, or `eal` (default: eal)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (auto-scaled if not specified)
- `--pfr-iter`: PFR iteration steps (default: 4)
- `--backbone-weights`: Path to pretrained ResNet18 weights (optional)
- `--output-dir`: Output directory for checkpoints
- `--resume`: Resume from checkpoint
- `--device`: Training device (auto-detected)

### Output

Training generates:
- `best_model.pth`: Best validation checkpoint
- `last_model.pth`: Final checkpoint  
- `training_*.csv`: Training logs


