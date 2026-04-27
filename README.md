# Multi-Modality Contrastive Representation Learning for Prostate Cancer Diagnosis

**BSc (Hons) Artificial Intelligence — Final Project Report | MOD002691**
**Anglia Ruskin University | April 2026**
**Author: Thejas Sunil (2118074) | Supervisor: Dr. Silvia Cirstea**

---

## Overview

This repository contains the full implementation of a multi-modality contrastive representation learning framework for binary classification of clinically significant prostate cancer (csPCa) using the [PI-CAI public dataset](https://zenodo.org/record/6624726).

The project investigates whether jointly pretraining MRI and clinical encoders in a shared embedding space — inspired by CLIP (Radford et al., 2021) — improves downstream cancer detection compared to supervised baselines and unimodal SimCLR pretraining. The best model achieves **0.8138 AUROC** on the 219-patient held-out test set.

---

## Key Results

| Model | Test AUROC | Cancer Recall |
|---|---|---|
| PSA Clinical Baseline (Random Forest) | 0.7100 | — |
| MRI Baseline (3D CNN) | 0.7102 | 0.541 |
| Fusion Baseline (MRI + PSA) | 0.7449 | 0.721 |
| SimCLR Fusion (unfrozen) | 0.7823 | — |
| **Cross-Modal Contrastive (unfrozen)** | **0.8138** | **0.770** |
| CLIP-B (SimCLR warm-start, unfrozen) | 0.8009 | — |
| CLIP-A (from scratch, unfrozen) | 0.7543 | — |
| biMRI + PSA Fusion | 0.7594 | — |

---

## Repository Structure

```
early_detection_of_prostate_cancer/
│
├── mri_baseline/
│   ├── models/
│   │   ├── mri_encoder.py          # 3D CNN encoder + MRIClassifier baseline
│   │   ├── psa_encoder.py          # Clinical MLP encoder + PSAClassifier baseline
│   │   └── fusion_model.py         # Multimodal fusion model (MRI + PSA)
│   ├── data/
│   │   └── multimodal_dataset.py   # Unified PyTorch dataset for all experiments
│   └── training/
│       ├── train_mri_baseline.py   # MRI-only supervised baseline training
│       ├── train_psa_baseline.py   # PSA-only supervised baseline training
│       └── train_fusion.py         # Supervised fusion training
│
├── contrastive/
│   ├── contrastive_dataset.py      # SimCLR dataset (two augmented views per scan)
│   ├── contrastive_model.py        # SimCLR model + NTXent loss
│   ├── crossmodal_dataset.py       # Cross-modal dataset (MRI + clinical per patient)
│   ├── crossmodal_model.py         # Cross-modal model + symmetric InfoNCE loss
│   ├── clip_model.py               # CLIP-style variants (Option A and B)
│   ├── train_contrastive.py        # SimCLR pretraining loop
│   ├── train_crossmodal.py         # Cross-modal contrastive pretraining loop
│   ├── train_clip.py               # CLIP-A and CLIP-B pretraining loop
│   ├── train_mri_contrastive.py    # SimCLR fine-tuning (MRI-only)
│   ├── train_fusion_contrastive.py # SimCLR fine-tuning (fusion)
│   └── train_fusion_crossmodal.py  # Cross-modal fine-tuning (fusion)
│
├── preprocessing/
│   ├── load_mri.py                 # Load, normalise and resize MRI volumes to .pt
│   ├── preprocess_clinical.py      # Clinical feature imputation and encoding
│   └── train_test_validation_split.py  # Stratified patient-level data split
│
├── explainability/
│   ├── gradcam_analysis.py         # Grad-CAM heatmap generation
│   └── shap_analysis.py            # SHAP clinical feature importance
│
└── README.md
```

---

## Methodology

### Data

- **Dataset:** PI-CAI public training dataset — 1,441 cases with complete mpMRI and clinical data
- **MRI sequences:** T2-weighted (T2W), apparent diffusion coefficient (ADC), high b-value DWI (HBV)
- **Clinical features:** PSA (ng/mL), PSAD (ng/mL²), prostate volume (mL), patient age (years)
- **Split:** 70% train (1,011) / 15% validation (219) / 15% test (219), stratified by csPCa label (~28% positive)

### Preprocessing

Each MRI volume is resampled to uniform spacing, normalised per-sequence (clip p1–p99, z-score), and saved as a `(3, 20, 160, 160)` float32 tensor. Clinical features are imputed algebraically (PSAD = PSA ÷ volume) and z-score normalised using training-set statistics only to prevent data leakage.

### Model Pipeline

```
Stage 1 — Baselines (supervised, from scratch)
    PSA Clinical Baseline    →  Random Forest on 4 clinical features
    MRI Baseline             →  3D CNN → 512-dim → classify
    Fusion Baseline          →  MRI (512) + PSA (256) → concat (768) → classify

Stage 2 — Contrastive Pretraining (self-supervised, no labels)
    SimCLR                   →  Two augmented MRI views → shared encoder → NTXent loss
    Cross-Modal              →  MRI + clinical pair → dual encoders → InfoNCE loss (CLIP-style)
    CLIP-A                   →  Cross-modal from scratch, wider 256-dim clinical encoder
    CLIP-B                   →  Cross-modal with SimCLR-pretrained MRI encoder warm-start

Stage 3 — Fine-tuning (supervised, with pretrained weights)
    Frozen                   →  Encoder weights fixed, only fusion head trained
    Unfrozen                 →  Full end-to-end fine-tuning with lower encoder LR
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA 12.x (tested on NVIDIA RTX 5090, 33.7 GB VRAM)
- PyTorch 2.x

### Setup

```bash
git clone https://github.com/tej1137/early_detection_of_prostate_cancer.git
cd early_detection_of_prostate_cancer
pip install -r requirements.txt
```

### Key dependencies

```
torch
torchio
SimpleITK
scikit-learn
numpy
pandas
matplotlib
seaborn
shap
```

---

## Usage

### 1. Preprocessing

```bash
# Split clinical data
python preprocessing/train_test_validation_split.py

# Preprocess clinical features (run for each split)
python preprocessing/preprocess_clinical.py \
    data/split_data/train_clinical.csv \
    data/preprocessed/train_preprocessed.csv

# Preprocess MRI volumes (run on GPU server)
python preprocessing/load_mri.py
```

### 2. Baseline Training

```bash
# MRI baseline
python -m mri_baseline.training.train_mri_baseline

# PSA clinical baseline
python -m mri_baseline.training.train_psa_baseline

# Fusion baseline
python -m mri_baseline.training.train_fusion
```

### 3. Contrastive Pretraining

```bash
# SimCLR unimodal MRI pretraining
python -m contrastive.train_contrastive

# Cross-modal contrastive pretraining (MRI + clinical)
python -m contrastive.train_crossmodal

# CLIP-style variants
python -m contrastive.train_clip --option_a   # from scratch
python -m contrastive.train_clip --option_b   # SimCLR warm-start
```

### 4. Fine-tuning

```bash
# SimCLR → MRI fine-tuning
python -m contrastive.train_mri_contrastive --unfrozen

# SimCLR → Fusion fine-tuning
python -m contrastive.train_fusion_contrastive --unfrozen

# Cross-modal → Fusion fine-tuning
python -m contrastive.train_fusion_crossmodal --unfrozen
```

### 5. Explainability

```bash
# Grad-CAM analysis on best model
python explainability/gradcam_analysis.py

# SHAP clinical feature importance
python explainability/shap_analysis.py
```

---

## Architecture Summary

### MRI Encoder (3D CNN)
Four sequential `Conv3DBlock` units (Conv3D → BN → ReLU → Conv3D → BN → ReLU → MaxPool) expanding channels from 3 → 32 → 64 → 128 → 256, followed by Global Average Pooling and a linear projection to **512-dim**. ~3.6M parameters. Used as backbone for both the MRI baseline classifier and all contrastive/fusion experiments.

### Clinical Encoder (MLP)
Three fully connected layers (4 → 64 → 128 → 256) with ReLU and dropout (0.2). Outputs a **256-dim** embedding. Used in fusion and cross-modal contrastive models.

### Cross-Modal Contrastive Model
Dual encoder framework adapting CLIP to the medical imaging domain. MRI and clinical encoders are jointly trained to align matched patient pairs in a shared **128-dim** embedding space using symmetric InfoNCE loss with learnable temperature (τ = 0.07). Projection heads (discarded after pretraining) map MRI: 512 → 256 → 128 and Clinical: 256 → 128 → 128.

---

## Explainability

**Grad-CAM** was applied to the final convolutional block of the best-performing cross-modal fusion model to visualise which prostate regions drove predictions. Activations were consistently located within the prostate gland, though one case showed contralateral activation relative to the expert-annotated lesion.

**SHAP** (KernelExplainer on 100 test cases) revealed clinical feature importance ranking: age > PSAD > PSA > volume. The unexpected age-first ordering likely reflects the demographics of the Dutch/German PI-CAI cohort.

---

## Infrastructure

All experiments were run on [RunPod](https://runpod.io) cloud GPU infrastructure with a single **NVIDIA RTX 5090 (33.7 GB VRAM)**. Data was stored in an S3 bucket integrated with RunPod.

- Cross-modal pretraining: ~63 minutes for 100 epochs
- Fine-tuning experiments: 15–30 minutes each
- Total GPU compute: ~34 hours across all experiments

---

## Citation

If you use this work please cite:

```
Sunil, T. (2026) Multi-Modality Contrastive Representation Learning for Prostate Cancer Diagnosis.
BSc Final Project Report, Anglia Ruskin University, MOD002691.
```

**Dataset:**
```
Bosma, J.S. et al. (2022) PI-CAI: Public Training and Development Dataset (v2.0).
Zenodo. doi: 10.5281/zenodo.6624726
```

---

## Acknowledgements

- PI-CAI consortium (Radboud University Medical Centre, Karolinska Institute, UMCG) for the public dataset
- RunPod for GPU infrastructure
- Dr. Silvia Cirstea for supervision and guidance

---

## License

This project is for academic purposes. The PI-CAI dataset is released under a Creative Commons license for research use. Please refer to the [PI-CAI dataset page](https://zenodo.org/record/6624726) for full licensing terms.