# 🧠 Brain Tumor Segmentation — 3D SegResNet + MONAI

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-0096FF)](https://monai.io/)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS%202021-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

A **full 3D volumetric segmentation** pipeline for brain tumor sub-region detection using the **SegResNet** architecture (MONAI) on the BraTS 2021 dataset. Unlike 2D slice-based approaches, this pipeline processes complete MRI volumes using patch-based training and sliding-window inference — predicting three clinically relevant sub-regions simultaneously.

---

## 📋 Table of Contents

- [Why This Matters](#why-this-matters)
- [Key Features](#key-features)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Advanced Techniques](#advanced-techniques)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Training Process](#training-process)
- [Performance & Metrics](#performance--metrics)
- [Future Improvements](#future-improvements)
- [References](#references)
- [License](#license)

---

## 🎯 Why This Matters

- **Global Impact**: Brain tumors affect ~300,000 people/year globally (WHO 2023)
- **Clinical Need**: Manual 3D segmentation takes 30–90 min per patient with high inter-observer variability
- **AI Solution**: Automated segmentation enables rapid, reproducible tumor delineation for treatment planning
- **Research Standard**: BraTS challenge is the international benchmark for brain tumor segmentation AI

---

## ✅ Key Features

- **SegResNet** — residual encoder-decoder built specifically for volumetric medical segmentation
- **3D patch-based training** — 128×128×128 patches, sliding-window inference at test time
- **Multi-class output** — simultaneous prediction of TC, WT, and ET sub-regions
- **EMA (Exponential Moving Average)** — shadow model for stable, higher-quality validation
- **Gradient accumulation** (×2) — simulates larger batch without extra GPU memory
- **Mixed-precision AMP** — ~2× faster training, ~50% less VRAM
- **Checkpoint resume** — continue from `checkpoint_epoch_025.pth` when a checkpoint dataset is attached
- **Rich CLI progress** — per-epoch table with TC/WT/ET Dice + timing
- **Enhanced visualisations** — dark-themed training dashboard with quality-colour-coded annotations and GT vs Prediction overlays with per-patient slice Dice display

---

## 📊 Results

### Validation Performance (notebook report)

| Sub-region | Description | Val Dice |
|-----------|-------------|----------|
| **TC** | Tumor Core (NCR + ET) | **0.9065** |
| **WT** | Whole Tumor (NCR + ED + ET) | **0.9283** |
| **ET** | Enhancing Tumor (ET only) | **0.8889** |
| **Mean** | Average across TC / WT / ET | **0.9078** |

> These are the best validation Dice values shown in the notebook figures.
> The notebook uses a 30-epoch base run and a separate checkpoint-resume cell for `checkpoint_epoch_025.pth`.

### Why ET is Hardest
- Enhancing Tumor is the **smallest** sub-region (often <5 cm³)
- Most critical for treatment planning — drives the need for high ET Dice
- 3D context (vs 2D slice methods) significantly improves ET detection

---

## 📂 Project Structure

```
brain-tumor-segmentation-3d-segresnet/
│
├── 📓 notebooks/
│   └── brain_tumor_segresnet_3d.ipynb   # Full 6-cell Kaggle pipeline
│
├── 📄 docs/
│   ├── pipeline.md                       # Stage-by-stage pipeline explanation
│   ├── architecture.md                    # SegResNet architecture deep-dive
│   └── assets/                           # Saved notebook figures
│
├── 🛠️ requirements.txt                   # Python dependencies
├── 🚫 .gitignore                          # Excludes data, models, caches
├── 📜 LICENSE                            # MIT License
└── 📖 README.md                          # This file
```

### Notebook Cells

| Cell | Purpose | Run order |
|------|---------|-----------|
| **Cell 0 — Kaggle bootstrap** | Kaggle environment preview / data listing | Before the main notebook |
| **Cell 1 — Install** | Install MONAI | 1st |
| **Cell 2 — Setup** | Imports, extraction, splits, transforms, model, optimiser | 2nd |
| **Cell 3 — Resume training** | Resume from `checkpoint_epoch_025.pth` for 5 more epochs | 3rd |
| **Cell 4 — Curves** | Training loss & Dice dashboard with quality annotations | After training |
| **Cell 5 — Visualisation** | GT vs Prediction overlays with per-patient slice Dice | After training |

---

## 📥 Dataset

**BraTS 2021 Task 1** — Brain Tumor Segmentation Challenge

| Property | Detail |
|----------|--------|
| Source | [Kaggle — BraTS 2021 Task 1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) |
| Subjects | ~1,251 training cases |
| Modalities | FLAIR, T1ce, T1, T2 (4 channels) |
| Format | NIfTI (`.nii.gz`) |
| Labels | 0=BG, 1=NCR, 2=ED, 4=ET → converted to TC / WT / ET |
| Split | 80% train / 10% val / 10% test (seed=42) |

### Label Convention

| Class | Sub-region | Composed of |
|-------|-----------|-------------|
| **TC** | Tumor Core | NCR (label 1) + ET (label 4) |
| **WT** | Whole Tumor | NCR + ED + ET (labels 1+2+4) |
| **ET** | Enhancing Tumor | Label 4 only |

### Data Structure (after extraction)
```
BraTS2021_Extracted/
└── BraTS2021_XXXXX/
    ├── BraTS2021_XXXXX_flair.nii.gz
    ├── BraTS2021_XXXXX_t1.nii.gz
    ├── BraTS2021_XXXXX_t1ce.nii.gz
    ├── BraTS2021_XXXXX_t2.nii.gz
    └── BraTS2021_XXXXX_seg.nii.gz
```

---

## 🏗️ Model Architecture

### SegResNet (MONAI)

```
Input:  [B, 4, 128, 128, 128]
        4-channel 3D patch (FLAIR · T1ce · T1 · T2)
              ↓
  ┌─────────────────────────────────────────────┐
  │  Residual Encoder                           │
  │  blocks_down = [1, 2, 2, 4]               │
  │  Stage 1: 32ch  (1 block)  stride→64³      │
  │  Stage 2: 64ch  (2 blocks) stride→32³      │
  │  Stage 3: 128ch (2 blocks) stride→16³      │
  │  Stage 4: 256ch (4 blocks) stride→8³       │
  └──────────────────┬──────────────────────────┘
                     ↓ skip connections ←→
  ┌─────────────────────────────────────────────┐
  │  Residual Decoder                           │
  │  blocks_up = [1, 1, 1]                     │
  │  Upsample back to 128³                      │
  └──────────────────┬──────────────────────────┘
                     ↓
Output: [B, 3, 128, 128, 128]
        3 independent binary masks: TC · WT · ET
```

| Parameter | Value |
|-----------|-------|
| Architecture | SegResNet |
| `blocks_down` | `[1, 2, 2, 4]` |
| `blocks_up` | `[1, 1, 1]` |
| `init_filters` | 32 |
| `in_channels` | 4 (FLAIR, T1ce, T1, T2) |
| `out_channels` | 3 (TC, WT, ET) |
| `dropout_prob` | 0.1 |
| Trainable params | ~4.8M |

### Why SegResNet over 3D U-Net?
- **Memory efficient**: strided convolutions instead of max-pooling, compact decoder
- **Residual connections**: prevents vanishing gradients on small lesions (ET)
- **All-3D operations**: no slice-by-slice approximation
- **MONAI-native**: sliding-window inference, medical transforms, Dice metric

---

## 🚀 Advanced Techniques

### 1. EMA (Exponential Moving Average)
```python
# After every optimiser step:
ema_param = 0.999 × ema_param + 0.001 × model_param
```
The EMA model is used **exclusively for validation** — smoother, more generalised predictions.

### 2. Sliding-Window Inference
```python
sliding_window_inference(
    inputs        = full_3d_volume,
    roi_size      = (128, 128, 128),
    sw_batch_size = 2,
    predictor     = ema_model,
    overlap       = 0.25,          # 25% overlap → Gaussian-weighted boundary blending
)
```
Required because full volumes (240×240×155) exceed GPU memory.

### 3. Loss — Dice + Focal
```python
loss = DiceLoss(sigmoid=True) + FocalLoss(gamma=2.0)
```
- **Dice**: directly optimises the evaluation metric; handles class imbalance
- **Focal (γ=2)**: down-weights easy background voxels; forces focus on hard tumour boundaries

### 4. Separate AdamW Parameter Groups
```python
# No weight-decay on biases and normalisation layers
optimizer = AdamW([
    {"params": decay_params,    "weight_decay": 1e-4},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=2e-4)
```
Correct L2 regularisation following AdamW best practices.

### 5. Full Configuration
```python
CFG = {
    "max_epochs"   : 30,
    "lr"           : 2e-4,        # AdamW peak LR
    "weight_decay" : 1e-4,
    "warmup_epochs": 5,            # Linear warmup
    "grad_accum"   : 2,            # Effective batch = 2
    "ema_decay"    : 0.999,
    "clip_grad"    : 1.0,
    "patience"     : 10,           # Early stopping
    "sw_overlap"   : 0.25,
    "ckpt_every"   : 3,            # Checkpoint every N epochs
}
```

### 6. MONAI Transform Pipeline

**Shared preprocessing (train + val + test)**
```
LoadImaged → EnsureChannelFirstd → ConvertToMultiChannelBasedOnBratsClassesd
    → Orientationd (RAS) → Spacingd (1×1×1 mm) → NormalizeIntensityd (z-score)
    → CropForegroundd (margin=10) → SpatialPadd (128³)
```

**Training augmentation only**
```
RandSpatialCropd → RandFlipd (3 axes) → RandRotate90d → RandZoomd [0.9–1.1]
    → RandGaussianNoised → RandScaleIntensityd → RandShiftIntensityd → RandAdjustContrastd
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (16 GB+ VRAM recommended for 3D volumes)
- Kaggle account (recommended) or local machine with 32+ GB RAM

### Option 1: Kaggle (Recommended)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) → **New Notebook**
2. Click **+ Add data** → search "BraTS 2021 Task 1" → Add
3. Upload `notebooks/brain_tumor_segresnet_3d.ipynb`
4. Settings → Accelerator → **GPU T4**
5. Run cells in order: Cell 0 → Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5

### Option 2: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/farhat-zenim/brain-tumor-segmentation-3d-segresnet.git
cd brain-tumor-segmentation-3d-segresnet

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
#    → https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
#    → Extract and update BASE_INPUT path in Cell 1

# 5. Launch notebook
jupyter notebook notebooks/brain_tumor_segresnet_3d.ipynb
```

---

## 📝 Usage

### Fresh Training
```python
# Cell 1: Run setup (no checkpoint in /kaggle/input/) → starts from scratch
# Cell 2: Runs 30 epochs, saves best_model.pth + periodic checkpoints
```

### Resuming Training
```python
# 1. Upload checkpoint_epoch_025.pth as a Kaggle Dataset
# 2. Attach it to your notebook
# 3. Run Cell 1 → Cell 3  — continues the checkpointed run for 5 epochs
```

### Inference on New Volumes
```python
import torch
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, NormalizeIntensityd, SpatialPadd, ToTensord,
)
from monai.transforms import Activations, AsDiscrete

# Load model
model = SegResNet(
    blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
    init_filters=32, in_channels=4, out_channels=3,
).cuda()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Preprocess
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(1, 1, 1), mode="bilinear"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    SpatialPadd(keys="image", spatial_size=(128, 128, 128)),
    ToTensord(keys="image"),
])

data   = transforms({"image": [flair_path, t1ce_path, t1_path, t2_path]})
volume = data["image"].unsqueeze(0).cuda()

# Infer
with torch.no_grad():
    pred  = sliding_window_inference(volume, (128, 128, 128), 2, model, overlap=0.25)
    masks = (torch.sigmoid(pred) > 0.5).cpu().numpy()  # (1, 3, D, H, W)
    tc, wt, et = masks[0, 0], masks[0, 1], masks[0, 2]
```

---

## 🧪 Training Process — Step by Step

1. **Install MONAI** — `pip install monai[all]`
2. **Extract dataset** — auto-discovers `.tar` archives under `/kaggle/input/`
3. **Build file list** — associates FLAIR/T1ce/T1/T2/seg for each patient
4. **Split data** — 80/10/10 train/val/test (seed=42, reproducible)
5. **Apply transforms** — orientation + spacing + normalisation + augmentation
6. **Build SegResNet** — 4→3 channels, 3D residual encoder-decoder
7. **Create EMA shadow** — copy of model for stable validation
8. **Configure loss** — Dice + Focal, AdamW + cosine annealing + warmup
9. **Train** — gradient accumulation, AMP, EMA update, rich progress display
10. **Validate** — sliding-window inference on EMA model, per-class Dice
11. **Save checkpoints** — every 3 epochs (only latest kept) + best model
12. **Visualise** — training curves dashboard + GT vs Prediction overlays

### Training Time

The notebook is configured for a 30-epoch base run plus an optional 5-epoch checkpoint-resume pass. Runtime depends on the GPU, MONAI version, and input throughput, so I am not pinning a fixed duration here.

---

## 📈 Performance & Metrics

### Reported Results (from the notebook)

| Metric | TC | WT | ET | Mean |
|--------|----|----|-----|------|
| **Val Dice** | 0.9065 | 0.9283 | 0.8889 | **0.9078** |

### Saved Figures

The main rendered outputs from the notebook are stored in [docs/assets](docs/assets):

- [training-curves-dashboard.png](docs/assets/training-curves-dashboard.png)
- [gt-vs-prediction-grid.png](docs/assets/gt-vs-prediction-grid.png)
- Raw notebook exports are kept alongside them as `notebook-output-*.png`.

### Checkpoint Structure
```python
{
    "epoch":                int,          # completed epoch
    "model_state_dict":     dict,         # live model weights
    "ema_state_dict":       dict,         # EMA model weights (used for validation)
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict":    dict,         # AMP GradScaler
    "best_dice":            float,        # best mean Dice seen so far
    "history":              dict,         # all metric lists
    "cfg":                  dict,         # full CFG dict for reproducibility
}
```

### Output Files

| File | Description |
|------|-------------|
| `best_model.pth` | EMA model weights at best validation Dice |
| `checkpoint_epoch_XXX.pth` | Latest periodic checkpoint (for resume) |
| `training_curves.png` | 6-panel training dashboard (loss + 4 Dice + summary) |
| `gt_vs_pred.png` | GT vs Prediction overlay grid with per-patient Dice |

---

## 🔑 Key Technical Skills Demonstrated

**Deep Learning**
- 3D volumetric segmentation with patch-based training
- SegResNet architecture (MONAI)
- Multi-class medical image segmentation
- EMA for training stability
- Sliding-window inference for large volumes
- Mixed-precision training (AMP)
- Dice + Focal loss for class-imbalanced medical data
- Proper AdamW with decoupled weight decay

**Engineering**
- Robust checkpoint save / auto-resume system
- Kaggle GPU memory management (batch=1, grad_accum=2)
- Automated dataset extraction from TAR archives
- Rich progress display (in-epoch bar + epoch table)
- Reproducible experiments (fixed seeds everywhere)
- Modular CFG-driven hyperparameter management

---

## 🚀 Future Improvements

**Model**
- [ ] Swin UNETR — transformer-based backbone for higher ceiling
- [ ] Test-time augmentation (TTA) — average predictions over flips
- [ ] Ensemble of SegResNet + SwinUNETR
- [ ] SWA (Stochastic Weight Averaging)

**Data & Training**
- [ ] CacheDataset — preload preprocessed volumes to RAM for faster training
- [ ] BraTS 2023 dataset — more recent benchmark
- [ ] K-fold cross-validation

**Deployment**
- [ ] Docker container for reproducible inference
- [ ] ONNX export for cross-platform deployment
- [ ] FastAPI REST endpoint
- [ ] Streamlit / Gradio inference UI
- [ ] 3D Slicer plugin integration

---

## ⚠️ Important Disclaimers

- This project is for **educational and research purposes only**
- Not clinically validated or FDA-approved
- Should **not** replace professional neuroradiological assessment
- Model performance may vary with different scanner protocols or institutions
- Always consult qualified medical professionals for clinical decisions

---

## 📚 References

1. **SegResNet**: Myronenko, A. (2019). 3D MRI brain tumor segmentation using autoencoder regularization. *MICCAI BrainLes Workshop*. [arXiv:1810.11654](https://arxiv.org/abs/1810.11654)

2. **BraTS 2021**: Baid, U., et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. [arXiv:2107.02314](https://arxiv.org/abs/2107.02314)

3. **MONAI**: MONAI Consortium. (2020). MONAI: Medical Open Network for AI. [GitHub](https://github.com/Project-MONAI/MONAI)

4. **Dice Loss**: Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV 2016*.

5. **Focal Loss**: Lin, T. Y., et al. (2017). Focal loss for dense object detection. *ICCV 2017*. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

6. **AdamW**: Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👥 Author & Acknowledgments

**Author**: Farhat Zenim

**Special Thanks**:
- The BraTS challenge organisers for the benchmark dataset
- The MONAI team for the medical imaging framework
- NVIDIA for the SegResNet architecture
- The Kaggle community for GPU resources and feedback

---

## 📧 Contact

- **GitHub**: [@farhat-zenim](https://github.com/farhat-zenim)
- **Project Link**: [brain-tumor-segmentation-3d-segresnet](https://github.com/farhat-zenim/brain-tumor-segmentation-3d-segresnet)

---

<div align="center">

**Built with ❤️ for medical AI**

⭐ Star this repo if you find it helpful!

[Report Bug](https://github.com/farhat-zenim/brain-tumor-segmentation-3d-segresnet/issues) ·
[Request Feature](https://github.com/farhat-zenim/brain-tumor-segmentation-3d-segresnet/issues)

</div>
