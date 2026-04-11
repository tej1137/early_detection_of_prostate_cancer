"""
bimri/data/bimri_dataset.py

Dataset for bi-parametric MRI (biMRI) experiments.

biMRI = T2W + ADC only  (drops HBV channel)
mpMRI = T2W + ADC + HBV (what you've been using so far)

Why biMRI matters clinically:
    HBV (high b-value diffusion) requires longer acquisition time and
    is not available in all centres. If biMRI achieves similar AUROC
    to mpMRI, it suggests a simpler, faster, cheaper protocol may
    be sufficient for csPCa detection.

    This is an active clinical research question — your results will
    directly contribute to this debate.

What changes vs multimodal_dataset.py:
    - MRI tensor loaded as (3, 20, 160, 160) from disk (always mpMRI)
    - We SLICE off channel 2 (HBV) → return (2, 20, 160, 160)
    - Everything else is identical

Why slice instead of re-preprocess:
    Your preprocessed .pt files already exist for all 1441 cases.
    Re-running preprocess_mri.py for 2 channels would take hours.
    Slicing [:2] at load time is instant and correct.

Channel order in your .pt files (from preprocess_mri.py):
    sequences = ["t2w", "adc", "hbv"]  → stacked in this order
    channel 0 = T2W
    channel 1 = ADC
    channel 2 = HBV  ← we drop this

Usage:
    from bimri.data.bimri_dataset import BiMRIDataset, get_bimri_loaders
"""

import json
import torch
import numpy as np
import pandas as pd
import torchio as tio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ══════════════════════════════════════════════════════════
# AUGMENTATION  (identical to multimodal_dataset.py)
# ══════════════════════════════════════════════════════════

def augment_bimri(tensor: torch.Tensor) -> torch.Tensor:
    """
    Same augmentation as mpMRI — applied to (2, D, H, W) tensor.
    torchio handles arbitrary channel counts so no changes needed.
    """
    t       = tensor.permute(0, 3, 2, 1)   # (2, W, H, D)
    subject = tio.Subject(mri=tio.ScalarImage(tensor=t))

    transform = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        tio.RandomAffine(scales=0.05, degrees=10, translation=5, p=0.5),
        tio.RandomNoise(std=(0, 0.05), p=0.3),
        tio.RandomBiasField(coefficients=0.3, p=0.3),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
    ])

    result = transform(subject).mri.tensor.permute(0, 3, 2, 1)
    return result.float().clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════

class BiMRIDataset(Dataset):
    """
    PyTorch Dataset for bi-parametric MRI (T2W + ADC only).

    Loads preprocessed .pt tensors (3 channels) and slices to 2 channels.
    Clinical features are NOT returned — biMRI experiment is MRI-only
    to give a clean comparison vs mpMRI-only baseline.

    Args:
        split    : "train", "val", or "test"
        augment  : apply augmentation (train split only)
        mri_dir  : path to preprocessed .pt MRI tensors
        csv_path : path to clinical_preprocessed.csv

    Returns per item:
        {
            "mri"    : FloatTensor (2, 20, 160, 160)  ← T2W + ADC only
            "label"  : LongTensor scalar (0 or 1)
            "case_id": str
        }
    """

    def __init__(
        self,
        split   : str,
        augment : bool = False,
        mri_dir : Path = Path("/workspace/data/preprocessed/mri"),
        csv_path: Path = Path("/workspace/data/preprocessed/clinical_preprocessed.csv"),
    ):
        self.split   = split
        self.augment = augment
        self.mri_dir = mri_dir

        # Load and filter clinical CSV for this split
        df       = pd.read_csv(csv_path, index_col="case_id")
        self.df  = df[df["split"] == split].copy()

        if len(self.df) == 0:
            raise ValueError(
                f"No cases found for split='{split}'. "
                f"Check csv_path: {csv_path}"
            )

        self.case_ids = self.df.index.tolist()

        print(
            f"  BiMRIDataset [{split}]: {len(self.case_ids)} cases  "
            f"({self.df['case_csPCa'].sum()} cancer, "
            f"{(self.df['case_csPCa'] == 0).sum()} benign)  "
            f"[T2W + ADC only, HBV dropped]"
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        case_id = self.case_ids[idx]
        label   = int(self.df.loc[case_id, "case_csPCa"])

        # Load full mpMRI tensor (3, 20, 160, 160)
        mri_path = self.mri_dir / f"{case_id}.pt"

        if not mri_path.exists():
            raise FileNotFoundError(
                f"Preprocessed MRI not found: {mri_path}\n"
                f"Run preprocess_mri.py first."
            )

        mri = torch.load(mri_path, weights_only=True)   # (3, 20, 160, 160)

        # ── DROP HBV — keep T2W (ch0) and ADC (ch1) only ──
        mri = mri[:2]   # (2, 20, 160, 160)

        # Apply augmentation (train only)
        if self.augment and self.split == "train":
            mri = augment_bimri(mri)

        return {
            "mri"    : mri,
            "label"  : torch.tensor(label, dtype=torch.long),
            "case_id": case_id,
        }


# ══════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ══════════════════════════════════════════════════════════

def get_bimri_loaders(
    batch_size  : int = 8,
    num_workers : int = 4,
) -> dict:
    """
    Build train / val / test DataLoaders for biMRI experiment.

    Uses WeightedRandomSampler on train to handle class imbalance
    (~72% benign, ~28% cancer) — same as train_mri_baseline.py.

    Returns:
        {
            "train": DataLoader,  — shuffled, augmented, weighted sampler
            "val"  : DataLoader,  — no shuffle, no augment
            "test" : DataLoader,  — no shuffle, no augment
        }
    """
    train_ds = BiMRIDataset("train", augment=True)
    val_ds   = BiMRIDataset("val",   augment=False)
    test_ds  = BiMRIDataset("test",  augment=False)

    # Weighted sampler — identical logic to train_mri_baseline.py
    labels       = [int(train_ds.df.loc[cid, "case_csPCa"])
                    for cid in train_ds.case_ids]
    class_counts = np.bincount(labels)
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )

    print(f"  Train class distribution — Benign: {class_counts[0]}  Cancer: {class_counts[1]}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    return {
        "train": train_loader,
        "val"  : val_loader,
        "test" : test_loader,
    }