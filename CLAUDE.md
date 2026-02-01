# CLAUDE.md — Semantic Guided Prototypical Networks (SGProtoNet)

## Project Overview

SGProtoNet is a few-shot medical image classification framework that enhances prototypical networks with semantic guidance from paired radiology reports. The core insight: prototypes computed from 1–5 examples are noisy; paired clinical text stabilizes and enriches them via cross-modal attention.

**Primary dataset**: IU Chest X-Ray (Indiana University) — ~7,500 chest X-ray images paired with radiology reports. Each sample has an image and a text report (findings + impressions) labeled with one or more pathology categories (cardiomegaly, effusion, infiltrate, no finding, etc.).

**Task formulation**: N-way K-shot episodic classification. Standard settings: 5-way 1-shot and 5-way 5-shot. Start single-label (dominant finding per image), extend to multi-label later.

---

## Architecture Components

### 1. Visual Encoder
- **Model**: ViT-B/16 (BiomedCLIP vision tower or CheXpert/MIMIC-CXR pretrained)
- **Outputs**: [CLS] token (768-d global feature) AND full patch token sequence (196 × 768 spatial features)
- **Default**: Frozen backbone. Fine-tune last N blocks only when explicitly enabled via config.

### 2. Semantic Encoder
- **Model**: PubMedBERT, BioBERT, or BiomedCLIP text tower
- **Input**: Preprocessed radiology report (findings + impressions, boilerplate stripped)
- **Outputs**: [CLS] token (768-d) and optionally full token sequence for cross-attention

### 3. Projection Heads
- Two separate MLPs mapping into a shared embedding space
- Visual: 768 → 512 → d (default d=256), with LayerNorm + GELU
- Semantic: 768 → 512 → d, same structure
- Patch tokens also projected: 768 → d per token

### 4. Semantic-Guided Attention Module (SGAM)
- Cross-attention where semantic embedding is the Query, visual patch tokens are Keys/Values
- Multi-head attention (default 4 heads)
- Output: semantically-attended visual feature v_guided ∈ ℝ^d
- This is the core contribution — language tells vision where to look

### 5. Gated Fusion
- Gate: g = σ(W_gate · [v_guided; s])
- Fused: g ⊙ v_guided + (1 − g) ⊙ s
- Final: MLP([v_cls; fused]) → ℝ^d
- Output is the multimodal representation f_final used for prototype computation

### 6. Prototype Computation
- Vanilla: p_c = mean(f_i) for support examples i of class c
- Enhanced: semantic-weighted aggregation using class-level description similarity as weights

### 7. Distance & Classification
- Support both Euclidean and Cosine distance (configurable, default cosine)
- p(y=c | x_q) = softmax(−d(f_q, p_c))

### 8. Inference Without Text (Three Strategies)
- **Class-level semantic anchors**: canonical text descriptions per pathology class (always available)
- **Visual-to-semantic projection**: auxiliary MLP predicting text embeddings from visual embeddings
- **Fallback**: visual-only mode using just v_cls (graceful degradation)
- Default: use class-level anchors; use per-image text when available

---

## Directory Structure

```
sgprotonet/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml          # Base config (all hyperparams with defaults)
│   ├── 5way_1shot.yaml        # Overrides for 5-way 1-shot
│   └── 5way_5shot.yaml        # Overrides for 5-way 5-shot
├── data/
│   ├── __init__.py
│   ├── iu_xray.py             # IU Chest X-Ray dataset loader
│   ├── preprocessing.py       # Report text cleaning, image transforms
│   ├── episode_sampler.py     # N-way K-shot episodic sampler
│   └── class_descriptions.py  # Canonical text descriptions per pathology class
├── models/
│   ├── __init__.py
│   ├── sgprotonet.py          # Full SGProtoNet model (top-level nn.Module)
│   ├── visual_encoder.py      # ViT wrapper, extracts [CLS] + patch tokens
│   ├── semantic_encoder.py    # Text encoder wrapper
│   ├── projections.py         # Visual and semantic projection heads
│   ├── sgam.py                # Semantic-Guided Attention Module
│   ├── fusion.py              # Gated fusion module
│   ├── prototypes.py          # Prototype computation (vanilla + semantic-weighted)
│   └── vis2sem.py             # Visual-to-semantic auxiliary projection
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Main training loop (two-phase)
│   ├── losses.py              # L_proto, L_align (InfoNCE), L_consist
│   ├── episode_loop.py        # Single episode forward/backward logic
│   └── schedulers.py          # LR scheduling, warmup
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py            # Meta-test evaluation over episodes
│   ├── metrics.py             # Accuracy, confidence intervals, per-class stats
│   └── visualize.py           # SGAM attention map visualization
├── scripts/
│   ├── train.py               # Entry point: python scripts/train.py --config configs/default.yaml
│   ├── eval.py                # Entry point: python scripts/eval.py --checkpoint <path>
│   ├── preprocess_iu_xray.py  # One-time data preprocessing script
│   └── ablation.py            # Run ablation experiments
└── tests/
    ├── test_sgam.py
    ├── test_episode_sampler.py
    ├── test_prototype.py
    └── test_fusion.py
```

---

## Tech Stack & Dependencies

- **Python**: 3.10+
- **Framework**: PyTorch 2.x
- **Pretrained models**: HuggingFace Transformers (ViT, PubMedBERT), open_clip (BiomedCLIP)
- **Config**: OmegaConf + YAML
- **Logging**: Weights & Biases (wandb) — optional, toggleable in config
- **Data**: torchvision transforms, pandas for metadata, nltk/regex for text preprocessing
- **Testing**: pytest

Install: `pip install torch torchvision transformers open_clip_torch omegaconf wandb pandas nltk pytest`

---

## Coding Conventions

### General
- Type hints on all function signatures. Use `torch.Tensor` not `Any` for tensor args.
- Docstrings on every public class and method (Google style).
- Modules are small and single-purpose. One concept per file.
- No wildcard imports. Explicit `from models.sgam import SGAM`.

### PyTorch Specific
- All model components inherit `nn.Module`.
- Use `@torch.no_grad()` for inference/evaluation methods.
- Shapes documented in comments: `# (batch, seq_len, d_model)`.
- Keep `forward()` methods clean — helper logic goes in private methods.
- Pretrained encoders wrapped in a class that exposes a uniform interface: `.encode(x) → (cls_token, full_sequence)`.

### Config
- All hyperparameters live in YAML configs, never hardcoded.
- Access via OmegaConf DictConfig objects.
- Every config key must have a default in `configs/default.yaml`.

### Naming
- Dimensions: `d_model` (shared embedding dim), `d_visual` (visual encoder dim), `d_semantic` (text encoder dim)
- Variables: `v_cls` (visual CLS), `v_patches` (patch tokens), `s_cls` (semantic CLS), `v_guided` (SGAM output), `f_final` (fused multimodal feature), `proto` (prototype vector)
- Files: snake_case. Classes: PascalCase. Constants: UPPER_SNAKE.

### Testing
- Every non-trivial module has a test file.
- Tests verify output shapes, gradient flow, and basic correctness.
- Run: `pytest tests/ -v`

---

## Training Details

### Two-Phase Training
**Phase 1 — Modality Alignment (non-episodic)**:
- Train projection heads + fusion modules on full dataset
- Loss: L_align (InfoNCE contrastive) between matched image-text pairs
- Encoders frozen
- Purpose: establish a well-structured shared embedding space

**Phase 2 — Episodic Meta-Training**:
- Switch to N-way K-shot episode sampling
- Loss: L_proto (cross-entropy on query classification) + λ₁·L_align + λ₂·L_consist
- Default: λ₁=0.5, λ₂=0.1
- Encoders frozen (optionally unfreeze last blocks via config)

### Data Splits
- Split BY CLASS, not by example. Example: 8 train / 3 val / 3 test classes.
- Meta-test classes are never seen during training.
- Defined in config, not hardcoded.

### Key Hyperparameters (defaults)
- `d_model`: 256
- `n_way`: 5
- `k_shot`: 1 (or 5)
- `q_query`: 15
- `sgam_heads`: 4
- `distance`: "cosine"
- `lr`: 1e-4 (AdamW)
- `episodes_per_epoch`: 500
- `num_epochs_phase1`: 20
- `num_epochs_phase2`: 100

---

## Common Commands

```bash
# Preprocess IU Chest X-Ray data
python scripts/preprocess_iu_xray.py --data_dir /path/to/iu_xray --output_dir data/processed/

# Train (full pipeline, both phases)
python scripts/train.py --config configs/5way_1shot.yaml

# Evaluate
python scripts/eval.py --checkpoint checkpoints/best.pt --config configs/5way_1shot.yaml

# Run ablation (removes one component at a time)
python scripts/ablation.py --config configs/5way_1shot.yaml

# Tests
pytest tests/ -v

# Visualize SGAM attention maps
python -m evaluation.visualize --checkpoint checkpoints/best.pt --image <path> --text "<report>"
```

---

## Implementation Order

Follow this sequence strictly. Each step should be validated before the next.

1. **Data pipeline**: `data/iu_xray.py`, `data/preprocessing.py`, `data/episode_sampler.py`
2. **Visual-only baseline**: `models/visual_encoder.py` + `models/prototypes.py` → basic ProtoNet (no text)
3. **Semantic encoder + projections**: `models/semantic_encoder.py`, `models/projections.py`, add L_align
4. **SGAM**: `models/sgam.py` → integrate into forward pass
5. **Gated fusion**: `models/fusion.py` → full multimodal prototype computation
6. **Class-level anchors + vis2sem**: `data/class_descriptions.py`, `models/vis2sem.py`
7. **Ablations and evaluation**: `scripts/ablation.py`, `evaluation/`

---

## Key Design Decisions to Remember

- **Never** average patch tokens into a single vector before SGAM — SGAM needs spatial structure.
- **Always** L2-normalize embeddings before cosine distance computation.
- **Text at test time is optional** — the pipeline must never crash if text is None. Use class-level anchors as fallback.
- **Prototype computation is differentiable** — gradients flow through it during meta-training.
- **Report preprocessing is not optional** — raw reports contain boilerplate that hurts performance. Always clean them.
- **Reproducibility**: seed everything (torch, numpy, random, CUDA). Episode sampling must be deterministic given a seed.

---

## What NOT to Do

- Do not fine-tune entire ViT/BERT from the start. Frozen first, selective unfreeze later.
- Do not use simple concatenation for fusion. Always use gated fusion.
- Do not hardcode class names or splits. Everything through config.
- Do not skip Phase 1 alignment training. The shared space quality matters.
- Do not evaluate on classes seen during training. Few-shot eval requires held-out classes.