# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGProtoNet (Semantic Guided Prototypical Networks) is a few-shot learning framework for medical image classification that combines visual and semantic (text) information. The model uses BiomedCLIP encoders and operates on chest X-ray images with radiology reports.

**Key Concept**: Few-shot episodic meta-learning where each training episode contains N classes (n_way) with K support examples (k_shot) and Q query examples (q_query) per class. The model learns to classify novel medical conditions from very few examples.

## Common Commands

### Setup and Installation
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies (includes pytest)
pip install -e ".[dev]"
```

### Data Preprocessing
```bash
# Preprocess IU Chest X-Ray dataset (one-time setup)
python scripts/preprocess_iu_xray.py \
  --data_dir /path/to/raw/iu_xray \
  --output_dir data/processed/
```

### Training
```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Train with experiment config (5-way 1-shot)
python scripts/train.py --config configs/5way_1shot.yaml

# Override config values from command line
python scripts/train.py --config configs/default.yaml \
  episode.k_shot=5 training.phase2.lr=5e-5
```

### Evaluation
```bash
# Evaluate a trained checkpoint on validation set
python scripts/eval.py --checkpoint checkpoints/best.pt --split val

# Evaluate on test set with config overrides
python scripts/eval.py --checkpoint checkpoints/best.pt --split test \
  inference.num_eval_episodes=1000
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_episode_sampler.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=.
```

## Architecture Overview

### Two-Phase Training

1. **Phase 1: Modality Alignment** (training/trainer.py:87-148)
   - Non-episodic training with InfoNCE contrastive loss
   - Aligns visual and semantic embeddings in shared space
   - Trains Vis2Sem module for text-free inference
   - Uses standard DataLoader with batch training

2. **Phase 2: Episodic Meta-Training** (training/trainer.py:150-267)
   - Episodic training with EpisodeSampler
   - Computes prototypes from support set, classifies queries
   - Combined loss: prototypical + alignment + consistency
   - Uses class anchor embeddings for text-free episodes

### Model Pipeline (models/sgprotonet.py)

The forward pass flows through these components:

1. **Visual Encoder** (models/visual_encoder.py) - Extracts [CLS] token and patch embeddings from images
2. **Semantic Encoder** (models/semantic_encoder.py) - Encodes text reports into semantic embeddings
3. **Projections** (models/projections.py) - Projects both modalities to shared d_model dimension
4. **SGAM** (models/sgam.py) - Semantic-Guided Attention Module: semantic embedding queries visual patches via cross-attention
5. **Gated Fusion** (models/fusion.py) - Fuses visual CLS, attended visual, and semantic features with learned gating
6. **Prototype Computation** (models/prototypes.py) - Averages support set features per class
7. **Distance & Classification** - Cosine or Euclidean distance to prototypes produces logits

### Text Strategies

The model supports three inference strategies (configs/default.yaml:103):

- `class_anchors`: Use pre-computed class description embeddings (default, best for few-shot)
- `vis2sem`: Generate semantic embedding from visual features via Vis2Sem module
- `visual_only`: Use visual features only (fallback)

### Episode Sampling (data/episode_sampler.py)

EpisodeSampler yields batches structured as:
```
[support_cls0 (k_shot), ..., support_clsN (k_shot),
 query_cls0 (q_query), ..., query_clsN (q_query)]
```

The `unpack_episode()` helper splits this into support/query sets and remaps labels to 0..N-1 within the episode.

## Configuration System

Uses OmegaConf with hierarchical YAML configs:

- `configs/default.yaml` - Base configuration with all defaults
- `configs/5way_1shot.yaml` - Override for 1-shot experiments
- `configs/5way_5shot.yaml` - Override for 5-shot experiments

Config merge order: default.yaml < experiment.yaml < CLI overrides

**Important config sections**:
- `model.*` - Architecture hyperparameters (d_model, encoder settings, SGAM, fusion)
- `episode.*` - Few-shot episode configuration (n_way, k_shot, q_query)
- `data.*` - Dataset paths and class splits (train_classes, val_classes, test_classes)
- `training.phase1.*` - Phase 1 hyperparameters
- `training.phase2.*` - Phase 2 hyperparameters with loss weights (lambda_align, lambda_consist)

## Key Implementation Details

### Class Splits
The dataset has limited samples per class. Class splits are defined in configs (configs/default.yaml:50-55):
- `train_classes`: Used for Phase 2 meta-training (at least 5 classes needed for 5-way)
- `val_classes`: Used for validation episodes
- `test_classes`: Used for final evaluation (may be empty if insufficient rare classes)

Each class must have at least `k_shot + q_query` samples per episode.

### Mixed Precision Training
AMP (Automatic Mixed Precision) is enabled by default (configs/default.yaml:60). Use `GradScaler` for backward pass (training/trainer.py:42,136,229).

### Checkpointing
- `checkpoint_dir`: Where checkpoints are saved (default: checkpoints/)
- `save_best`: Saves best.pt when validation accuracy improves
- `save_every_n_epochs`: Periodic checkpoint saving

Checkpoints include: model_state_dict, config, epoch, accuracy.

### Loss Functions (training/losses.py)

- **Prototypical Loss**: Cross-entropy between query logits and labels
- **Alignment Loss**: InfoNCE contrastive loss between visual and semantic embeddings
- **Consistency Loss**: MSE between fused features and visual-only features
- **Vis2Sem Loss**: MSE for visual-to-semantic prediction

## File Organization

```
sgprotonets/
├── configs/          # YAML configuration files
├── data/             # Dataset loaders and episode sampling
│   ├── iu_xray.py           # IU X-Ray dataset class
│   ├── episode_sampler.py   # N-way K-shot episode sampler
│   ├── class_descriptions.py # Medical finding descriptions
│   └── preprocessing.py      # Image transforms, text cleaning
├── models/           # Neural network modules
│   ├── sgprotonet.py        # Top-level model
│   ├── visual_encoder.py    # BiomedCLIP vision encoder
│   ├── semantic_encoder.py  # BiomedCLIP text encoder
│   ├── sgam.py              # Semantic-Guided Attention Module
│   ├── fusion.py            # Gated fusion
│   ├── projections.py       # Modality projection heads
│   ├── prototypes.py        # Prototype computation
│   └── vis2sem.py           # Visual-to-semantic mapping
├── training/         # Training infrastructure
│   ├── trainer.py           # Two-phase trainer
│   ├── episode_loop.py      # Single episode forward/loss
│   ├── losses.py            # Loss functions
│   └── schedulers.py        # LR schedulers
├── evaluation/       # Evaluation and metrics
│   ├── evaluate.py          # Meta-test evaluation
│   └── metrics.py           # Classification metrics
├── scripts/          # Entry points
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation script
│   └── preprocess_iu_xray.py # Data preprocessing
└── tests/            # Unit tests
```

## Working with This Codebase

### Adding New Datasets
1. Create dataset class in `data/` following IUXRayDataset pattern (data/iu_xray.py:16-80)
2. Ensure dataset provides `class_indices` dict for EpisodeSampler
3. Add preprocessing script in `scripts/`
4. Update config with new class splits

### Modifying the Model
- Visual/semantic encoders support BiomedCLIP, ViT, PubMedBERT, BioBERT (check type in config)
- SGAM uses cross-attention: semantic as Query, visual patches as Key/Value
- Distance metric (cosine vs euclidean) is configurable in model config
- Prototype mode: "vanilla" (mean pooling) or "semantic_weighted" (weighted by class anchors)

### Debugging Episodes
Use `unpack_episode()` to inspect episode structure. Labels are remapped to 0..N-1 within each episode, so model predictions are class indices within the episode, not global class IDs.

### Hyperparameter Tuning
Key hyperparameters for few-shot performance:
- `episode.k_shot`: Number of support examples (1, 5, 10)
- `training.phase2.lambda_align`: Alignment loss weight (0.1-1.0)
- `training.phase2.lambda_consist`: Consistency loss weight (0.05-0.2)
- `model.d_model`: Shared embedding dimension (128-512)
- `model.sgam.num_heads`: Attention heads in SGAM

### WandB Logging
Enable with `logging.use_wandb=true` and set `logging.wandb_project` and `logging.wandb_entity` in config.
