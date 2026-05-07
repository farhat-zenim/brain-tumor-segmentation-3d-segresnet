# Architecture — SegResNet for 3D Brain Tumor Segmentation

## Overview

SegResNet (Myronenko, NVIDIA 2019) is a residual encoder-decoder architecture designed specifically for volumetric medical image segmentation. It was originally proposed for the BraTS challenge and achieves state-of-the-art results on 3D MRI segmentation tasks.

**Paper**: [3D MRI Brain Tumor Segmentation Using Autoencoder Regularization](https://arxiv.org/abs/1810.11654)

---

## Why SegResNet over other architectures?

| Architecture | 3D Native | Memory | BraTS SOTA | Notes |
|-------------|-----------|--------|-----------|-------|
| **SegResNet** | ✅ | Efficient | ✅ | Purpose-built for this task |
| 3D U-Net | ✅ | High | ✗ | Simpler, less expressive |
| SwinUNETR | ✅ | Very high | ✅ | Transformer — needs 24+ GB |
| 2D U-Net (slice) | ✗ | Low | ✗ | Loses inter-slice context |
| nnU-Net | ✅ | High | ✅ | Auto-config, less flexible |

SegResNet balances **expressiveness**, **memory efficiency**, and **proven BraTS performance** — ideal for Kaggle T4 (16 GB).

---

## Architecture Details

### Input
- Shape: `[Batch, 4, 128, 128, 128]`
- 4 MRI modalities: FLAIR · T1ce · T1 · T2

### Encoder

The encoder follows a hierarchical downsampling path with residual blocks:

```
Input (4ch)
    ↓  Conv 3×3×3, 32 filters
Stage 1: 32ch  — 1 residual block  — stride 2 → 64×64×64
Stage 2: 64ch  — 2 residual blocks — stride 2 → 32×32×32
Stage 3: 128ch — 2 residual blocks — stride 2 → 16×16×16
Stage 4: 256ch — 4 residual blocks — stride 2 →  8×8×8
```

Each **residual block**:
```
x → GroupNorm → ReLU → Conv 3×3×3 → GroupNorm → ReLU → Conv 3×3×3 → + x
```

### Decoder

Symmetric upsampling path with skip connections from encoder:

```
Stage 4:  8×8×8  ×256 → Deconv + Cat(skip) → 16×16×16 ×128
Stage 3: 16×16×16 ×128 → Deconv + Cat(skip) → 32×32×32 ×64
Stage 2: 32×32×32 ×64  → Deconv + Cat(skip) → 64×64×64 ×32
```

### Head
```
64×64×64 × 32 → Conv 1×1×1 → 3 outputs → Sigmoid → Binary masks
```

### Output
- Shape: `[Batch, 3, 128, 128, 128]`
- 3 independent binary probability maps: TC · WT · ET

---

## EMA (Exponential Moving Average)

A shadow copy of the model maintains a running average of weights:

```python
ema_weight = decay * ema_weight + (1 - decay) * model_weight
# decay = 0.999 — slower update (roughly 1,000 step half-life)
```

**Why EMA for validation?**
- Smoother weight trajectory → less noisy Dice scores
- Better generalisation — less sensitive to recent gradient noise
- Standard practice in medical imaging (nnU-Net, BraTS winners)

The EMA model is **only used for validation and inference** — training gradients flow through the live model only.

---

## Loss Functions

### Dice Loss
```python
L_dice = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
```
- Directly optimises the evaluation metric
- Robust to class imbalance (background vs tumour)
- `smooth_nr=1e-5, smooth_dr=1e-5` — numerically stable

### Focal Loss
```python
L_focal = -α_t × (1 - p_t)^γ × log(p_t)    # γ = 2.0
```
- Down-weights easy background voxels (already classified correctly)
- Forces the model to focus on hard boundary voxels
- Critical for small ET region

### Combined Loss
```python
L_total = L_dice + L_focal
```
Equal weighting — both losses operate on the same scale (0–1).

---

## Optimiser & Scheduler

### AdamW
- Decoupled weight decay (correct L2 regularisation)
- No weight decay on biases and normalisation layers (separate param groups)
- `lr=2e-4`, `weight_decay=1e-4`, `betas=(0.9, 0.999)`

### Learning Rate Schedule
```
Epochs 1–5:    Linear warmup   (0.1 × lr → lr)
Epochs 6–30:   Cosine annealing (lr → eta_min=1e-6)
```

Warmup prevents early training instability when patch-normalised volumes have high variance.

---

## Sliding-Window Inference

Full 3D volumes cannot be processed in a single forward pass (memory). MONAI's `sliding_window_inference` tiles the volume with overlapping patches:

1. Extract overlapping 128³ patches (25% overlap)
2. Run model forward pass on each patch
3. Accumulate predictions weighted by Gaussian kernel (smooth at boundaries)
4. Normalise by weight sum

**Overlap=0.25**: each voxel is covered by ~1.95 patches on average — good trade-off between quality and speed.

---

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| Encoder (stem + 4 stages) | ~3.1M |
| Decoder (3 stages + head) | ~1.7M |
| **Total** | **~4.8M** |

Compact enough for single T4 GPU training with 3D 128³ patches.
