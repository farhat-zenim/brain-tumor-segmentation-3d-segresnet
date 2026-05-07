# Pipeline — 3D SegResNet Brain Tumor Segmentation

This document describes every stage of the pipeline in detail.

---

## Stage 1 — Data Discovery & Extraction

The notebook scans `/kaggle/input/` recursively for any `.tar` or `.tar.gz` archive and extracts it to `/kaggle/working/BraTS2021_Extracted/`.

After extraction, valid case directories are identified as **folders containing ≥ 5 `.nii.gz` files** (FLAIR, T1, T1ce, T2, seg).

**Split (random seed=42):**

| Subset | Fraction | Cases (~1,251 total) | Purpose |
|--------|----------|----------------------|---------|
| Train | 80% | ~1,001 | Gradient updates |
| Validation | 10% | ~125 | Epoch-level Dice evaluation |
| Test | 10% | ~125 | Final held-out evaluation |

---

## Stage 2 — Label Convention

BraTS 2021 raw segmentation labels:

| Raw value | Region |
|-----------|--------|
| 0 | Background |
| 1 | Necrotic core (NCR) |
| 2 | Peritumoral edema (ED) |
| 4 | Enhancing tumor (ET) |

MONAI's `ConvertToMultiChannelBasedOnBratsClassesd` converts to 3-channel binary masks:

| Channel | Sub-region | Composition |
|---------|-----------|-------------|
| 0 | TC — Tumor Core | Labels 1 + 4 |
| 1 | WT — Whole Tumor | Labels 1 + 2 + 4 |
| 2 | ET — Enhancing Tumor | Label 4 only |

---

## Stage 3 — MONAI Transform Pipeline

### Shared Preprocessing (train + val + test)

```
LoadImaged                              → Load NIfTI volumes from disk
EnsureChannelFirstd                     → Add channel dimension to image
ConvertToMultiChannelBasedOnBratsClassesd → Convert label → 3-channel binary
Orientationd (axcodes="RAS")            → Standardise anatomical orientation
Spacingd (pixdim=1.0×1.0×1.0 mm)       → Resample to isotropic 1 mm voxels
NormalizeIntensityd (nonzero, ch_wise)  → Z-score normalise per channel (brain only)
CropForegroundd (margin=10)             → Remove zero-background padding
SpatialPadd (spatial_size=128³)         → Pad to minimum patch size
```

### Training Augmentation (train only)

```
RandSpatialCropd (128³)                 → Random 3D patch extraction
RandFlipd (axis=0, p=0.5)              → Left-right flip
RandFlipd (axis=1, p=0.5)              → Anterior-posterior flip
RandFlipd (axis=2, p=0.5)              → Inferior-superior flip
RandRotate90d (p=0.5, max_k=3)         → Random 90° rotation
RandZoomd [0.9–1.1] (p=0.3)           → Random mild zoom
RandGaussianNoised (std=0.05, p=0.25)  → Additive Gaussian noise
RandScaleIntensityd (factor=0.1, p=0.3) → Multiplicative intensity jitter
RandShiftIntensityd (offset=0.1, p=0.3) → Additive intensity shift
RandAdjustContrastd (γ=0.7–1.5, p=0.3) → Gamma contrast adjustment
ToTensord                               → Convert to PyTorch tensors
```

---

## Stage 4 — Model: SegResNet

**SegResNet** (Myronenko 2019) is a residual encoder-decoder designed for volumetric medical image segmentation.

```
[Input: 4×128×128×128]
       ↓
[Conv stem: 32 filters]
       ↓
[Encoder Stage 1: 32ch  → ×1 res block,  stride 2 → 64³] ───────── skip →
[Encoder Stage 2: 64ch  → ×2 res blocks, stride 2 → 32³] ───── skip →
[Encoder Stage 3: 128ch → ×2 res blocks, stride 2 → 16³] ─── skip →
[Encoder Stage 4: 256ch → ×4 res blocks, stride 2 → 8³ ] ─ skip →
                                                                       │
[Decoder Stage 3: 128ch ← Deconv + Cat(skip)] ←──────────────────────┘
[Decoder Stage 2: 64ch  ← Deconv + Cat(skip)] ←──────────────────────┘
[Decoder Stage 1: 32ch  ← Deconv + Cat(skip)] ←──────────────────────┘
       ↓
[Output conv 1×1×1: 3ch → Sigmoid]
[Output: 3×128×128×128]
```

**EMA shadow model** maintains a moving average of weights for validation:
```
ema_w = 0.999 × ema_w + 0.001 × model_w
```
This produces smoother, more generalised predictions.

**Total parameters: ~4.8M** — compact enough for single T4 GPU (16 GB) training.

---

## Stage 5 — Training Loop

### Per-Step
1. Load batch (1 patch of 128³)
2. Forward pass with `autocast()` (AMP FP16)
3. Compute `DiceLoss + FocalLoss`
4. Scale loss by `1 / grad_accum` (=2)
5. Backward pass (accumulate gradients)
6. Every `grad_accum` steps: unscale → clip gradients (norm=1.0) → optimizer step → EMA update

### Per-Epoch
1. Complete training pass over all patches
2. Update LR scheduler:
       - Epochs 1–5: Linear warmup (0.1×lr → lr)
       - Epochs 6–30: CosineAnnealingLR (lr → 1e-6)
3. Validate on EMA model with sliding-window inference
4. Record mean Dice (TC, WT, ET)
5. Save `best_model.pth` if mean Dice improved
6. Save periodic checkpoint every 3 epochs (latest only — saves disk space)
7. Optional resume cell can load `checkpoint_epoch_025.pth` and continue the run for 5 more epochs

### Loss Function
```
L_total = L_dice(sigmoid=True, batch=True) + L_focal(γ=2.0)
```
- `L_dice`: directly optimises the evaluation metric; robust to class imbalance
- `L_focal`: focuses learning on hard boundary voxels (especially small ET)

---

## Stage 6 — Sliding-Window Inference

Full 3D volumes (≈240×240×155 after resampling) do not fit in GPU memory for a single forward pass. MONAI's `sliding_window_inference` tiles them with overlapping patches:

```python
sliding_window_inference(
    inputs        = full_volume,     # [1, 4, D, H, W]
    roi_size      = (128, 128, 128),
    sw_batch_size = 2,               # process 2 patches at once
    predictor     = ema_model,
    overlap       = 0.25,            # 25% overlap — Gaussian-weighted blending
)
```

Each voxel is covered by ~1.95 overlapping patches (at overlap=0.25) — good trade-off between quality and speed.

---

## Stage 7 — Checkpointing

Checkpoints are saved every 3 epochs; only the **latest** is kept to respect Kaggle's disk limit.

```python
checkpoint = {
    "epoch":                int,          # completed epoch
    "model_state_dict":     dict,         # live model weights
    "ema_state_dict":       dict,         # EMA model weights
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict":    dict,         # AMP GradScaler state
    "best_dice":            float,
    "history":              dict,         # all metric lists
    "cfg":                  dict,         # full CFG for reproducibility
}
```

**Resume workflow:** A dedicated notebook cell expects `checkpoint_epoch_025.pth` to be attached as a Kaggle Dataset, then loads model, EMA, optimizer, scheduler, scaler, history, and continues for 5 epochs.

---

## Stage 8 — Metrics

MONAI `DiceMetric` with `reduction="mean_batch"` computes per-class Dice over the full validation set:

| Metric | Description | Best value shown in notebook |
|--------|-------------|-------------------|
| `val_tc` | Dice for Tumor Core | 0.9065 |
| `val_wt` | Dice for Whole Tumor | 0.9283 |
| `val_et` | Dice for Enhancing Tumor | 0.8889 |
| `val_dice` | Mean of TC + WT + ET | **0.9078** |

Best model selected by **mean Dice** and saved as `best_model.pth`.

---

## Stage 9 — Outputs

| File | Description |
|------|-------------|
| `best_model.pth` | EMA model weights at best validation Dice |
| `checkpoint_epoch_XXX.pth` | Latest periodic checkpoint (for resume) |
| `docs/assets/training-curves-dashboard.png` | 6-panel dashboard: loss + 4 Dice curves + summary panel |
| `docs/assets/gt-vs-prediction-grid.png` | GT vs Prediction overlay grid with per-patient slice Dice |
| `docs/assets/notebook-output-*.png` | Raw exported notebook figures kept for traceability |
