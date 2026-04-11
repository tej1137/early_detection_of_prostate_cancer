"""
contrastive/crossmodal_dataset.py

Dataset for cross-modal contrastive pretraining.

Each item returns ONE patient's:
    - MRI tensor     (augmented)         → (3, 20, 160, 160)
    - Clinical vector (PSA, PSAD, vol, age) → (4,)

The cross-modal contrastive loss then pulls:
    MRI_i  ↔  Clinical_i   (same patient)  → POSITIVE pair
    MRI_i  ↔  Clinical_j   (diff patient)  → NEGATIVE pair

This is fundamentally different from SimCLR:
    SimCLR:      MRI_aug1  ↔  MRI_aug2    (same scan, two views)
    Cross-modal: MRI       ↔  Clinical    (same patient, two modalities)

Why this matters:
    The model learns that a patient's MRI appearance and their
    clinical biomarkers are two views of the same underlying biology.
    After training, both encoders live in a shared latent space where
    MRI and clinical embeddings for the same patient are close.

Usage:
    from contrastive.crossmodal_dataset import CrossModalDataset, get_crossmodal_loader
    loader = get_crossmodal_loader()
"""

import json
import torch
import numpy as np
import pandas as pd
import torchio as tio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════
# AUGMENTATION — applied to MRI only, not clinical features
# Clinical features are deterministic (no augmentation needed)
# ══════════════════════════════════════════════════════════

def get_mri_augmentation():
    """
    Moderate augmentation for cross-modal pretraining.

    Lighter than SimCLR (which uses two heavy augmentations of the same
    scan). Here we apply one augmentation per scan per epoch, since the
    cross-modal signal comes from the MRI↔clinical alignment, not from
    two MRI views of the same scan.
    """
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        tio.RandomAffine(scales=0.08, degrees=12, translation=6, p=0.5),
        tio.RandomNoise(std=(0, 0.06), p=0.35),
        tio.RandomBiasField(coefficients=0.35, p=0.35),
        tio.RandomGamma(log_gamma=(-0.35, 0.35), p=0.35),
        tio.RandomBlur(std=(0, 0.4), p=0.2),
    ])


# ══════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════

class CrossModalDataset(Dataset):
    """
    Returns paired (MRI, clinical) for each patient.

    Both modalities are for the SAME patient — this is the positive pair.
    Negative pairs are formed in-batch by the loss function.

    Args:
        splits    : list of splits to include, e.g. ["train"] or ["train","val","test"]
        mri_dir   : path to preprocessed .pt MRI tensors
        csv_path  : path to clinical_preprocessed.csv
        stats_path: path to norm_stats.json (for clinical normalisation)
        augment   : whether to augment MRI (True for pretraining)

    Returns per item:
        {
            "mri"     : FloatTensor (3, 20, 160, 160)   — augmented MRI
            "clinical": FloatTensor (4,)                — normalised [psa, psad, vol, age]
            "case_id" : str                              — for debugging
        }
    """

    # Clinical feature order — must match preprocess_clinical.py
    CLINICAL_FEATURES = ["psa", "psad", "prostate_volume", "patient_age"]

    def __init__(
        self,
        splits    : list = ["train", "val", "test"],
        mri_dir   : Path = Path("/workspace/data/preprocessed/mri"),
        csv_path  : Path = Path("/workspace/data/preprocessed/clinical_preprocessed.csv"),
        stats_path: Path = Path("/workspace/data/preprocessed/norm_stats.json"),
        augment   : bool = True,
    ):
        self.mri_dir = mri_dir
        self.augment = augment
        self.transform = get_mri_augmentation() if augment else None

        # ── Load clinical CSV ──────────────────────────────
        df = pd.read_csv(csv_path, index_col="case_id")
        self.df = df[df["split"].isin(splits)].copy()

        if len(self.df) == 0:
            raise ValueError(
                f"No cases found for splits={splits}. "
                f"Check csv_path: {csv_path}"
            )

        self.case_ids = self.df.index.tolist()

        # ── Load normalisation stats ───────────────────────
        with open(stats_path, "r") as f:
            stats = json.load(f)

        # ── Pre-normalise all clinical features ───────────
        # Shape: (N, 4) — done once at init, not per-item
        features = []
        for col in self.CLINICAL_FEATURES:
            mean = stats[col]["mean"]
            std  = stats[col]["std"]
            norm = (self.df[col].values - mean) / std
            features.append(norm)

        self.clinical_tensor = torch.tensor(
            np.stack(features, axis=1),
            dtype=torch.float32
        )   # (N, 4)

        print(
            f"  CrossModalDataset: {len(self.case_ids)} patients "
            f"(splits={splits}, augment={augment})"
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        case_id = self.case_ids[idx]

        # ── MRI ───────────────────────────────────────────
        mri_path = self.mri_dir / f"{case_id}.pt"
        mri = torch.load(mri_path, weights_only=True)   # (3, 20, 160, 160)

        if self.augment:
            mri = self._augment_mri(mri)

        # ── Clinical ──────────────────────────────────────
        clinical = self.clinical_tensor[idx]   # (4,)

        return {
            "mri"     : mri,
            "clinical": clinical,
            "case_id" : case_id,
        }

    def _augment_mri(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply torchio augmentation to a (3, D, H, W) tensor."""
        # torchio expects (C, W, H, D) — permute in and back out
        t = tensor.permute(0, 3, 2, 1)
        subject = tio.Subject(mri=tio.ScalarImage(tensor=t))
        result  = self.transform(subject).mri.tensor
        return result.permute(0, 3, 2, 1).float().clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ══════════════════════════════════════════════════════════

def get_crossmodal_loader(
    batch_size  : int  = 32,
    num_workers : int  = 4,
    splits      : list = ["train", "val", "test"],
    augment     : bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for cross-modal contrastive pretraining.

    Uses ALL splits by default — no labels are used so there's no
    leakage risk. More data = more negatives in each batch = better
    contrastive learning signal.

    Args:
        batch_size : 32 works well on RTX 5090 (32GB VRAM)
        num_workers: 4 is safe for RunPod
        splits     : which splits to include
        augment    : whether to apply MRI augmentation

    Returns:
        DataLoader yielding {"mri", "clinical", "case_id"} dicts
    """
    dataset = CrossModalDataset(splits=splits, augment=augment)

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,   # InfoNCE needs full batches for consistent negatives
    )