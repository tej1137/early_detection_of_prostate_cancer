"""
data/multimodal_dataset.py

PyTorch Dataset for PI-CAI preprocessed data.

Loads:
  - MRI tensor from /workspace/data/preprocessed/mri/{case_id}.pt
  - Clinical features from clinical_preprocessed.csv
  - Normalisation stats from norm_stats.json

Returns per case:
  {
    "mri":      FloatTensor (3, 20, 160, 160),
    "clinical": FloatTensor (4,),               ← [psa, psad, volume, age] normalised
    "label":    LongTensor  scalar (0 or 1),
    "case_id":  str
  }

Usage:
  from mri_baseline.data.multimodal_dataset import get_dataloaders
  loaders = get_dataloaders(config)
  train_loader = loaders["train"]
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchio as tio


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

class DataConfig:
    # ── Preprocessed data ─────────────────────────────────
    clinical_csv  = Path("/workspace/data/preprocessed/clinical_preprocessed.csv")
    norm_stats    = Path("/workspace/data/preprocessed/norm_stats.json")
    mri_dir       = Path("/workspace/data/preprocessed/mri")

    # ── Clinical features (must match preprocess_clinical.py) ──
    clinical_features = ["psa", "psad", "prostate_volume", "patient_age"]
    target_col        = "case_csPCa"

    # ── DataLoader settings ────────────────────────────────
    batch_size        = 8
    num_workers       = 4
    pin_memory        = True    # faster GPU transfer

    # ── Augmentation (train split only) ───────────────────
    augment_train     = True


# ══════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════

def augment_mri(tensor: torch.Tensor) -> torch.Tensor:
    """
    Enhanced MRI augmentation using torchio.
    tensor shape: (3, D, H, W) → (3, D, H, W)
    """
    # torchio expects (C, W, H, D) — permute in and back out
    t = tensor.permute(0, 3, 2, 1)   # (3, W, H, D)

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
# DATASET CLASS
# ══════════════════════════════════════════════════════════

class PiCAIDataset(Dataset):
    """
    PyTorch Dataset for PI-CAI preprocessed MRI + clinical data.

    Args:
        split       : "train", "val", or "test"
        config      : DataConfig instance
        augment     : If True, apply augmentation (train only)
        mri_only    : If True, skip clinical features (for MRI-only model)
        clinical_only: If True, skip MRI loading (for PSA-only model, fast!)
    """

    def __init__(
        self,
        split          : str,
        config         : DataConfig = None,
        augment        : bool = False,
        mri_only       : bool = False,
        clinical_only  : bool = False,
    ):
        self.config        = config or DataConfig()
        self.split         = split
        self.augment       = augment
        self.mri_only      = mri_only
        self.clinical_only = clinical_only

        # Load clinical CSV and filter to this split
        df = pd.read_csv(self.config.clinical_csv, index_col="case_id")
        self.df = df[df["split"] == split].copy()

        if len(self.df) == 0:
            raise ValueError(f"No cases found for split='{split}' in {self.config.clinical_csv}")

        self.case_ids = self.df.index.tolist()

        # Load normalisation stats (computed on train set only)
        with open(self.config.norm_stats, 'r') as f:
            self.norm_stats = json.load(f)

        # Precompute normalised clinical features as numpy array
        # Shape: (N, 4) — faster than normalising per-item in __getitem__
        self._precompute_clinical()

        print(f"  PiCAIDataset [{split}]: {len(self.case_ids)} cases  "
              f"({self.df[self.config.target_col].sum()} cancer, "
              f"{(self.df[self.config.target_col] == 0).sum()} benign)")

    def _precompute_clinical(self):
        """
        Normalise all clinical features upfront using train-set stats.

        Formula: z = (x - mean) / std

        Using train-set mean/std for ALL splits (train, val, test)
        to prevent data leakage — val/test never influence the scale.
        """
        features = []
        for col in self.config.clinical_features:
            mean = self.norm_stats[col]["mean"]
            std  = self.norm_stats[col]["std"]
            normalised = (self.df[col].values - mean) / std
            features.append(normalised)

        # Stack → (N, 4), then convert to tensor
        self.clinical_tensor = torch.tensor(
            np.stack(features, axis=1),
            dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        case_id = self.case_ids[idx]
        label   = int(self.df.loc[case_id, self.config.target_col])

        result = {
            "label"   : torch.tensor(label, dtype=torch.long),
            "case_id" : case_id,
        }

        # ── Clinical features ──────────────────────────────
        if not self.mri_only:
            result["clinical"] = self.clinical_tensor[idx]   # (4,)

        # ── MRI tensor ─────────────────────────────────────
        if not self.clinical_only:
            mri_path = self.config.mri_dir / f"{case_id}.pt"

            if not mri_path.exists():
                raise FileNotFoundError(
                    f"Preprocessed MRI not found: {mri_path}\n"
                    f"Run preprocess_mri.py first."
                )

            mri = torch.load(mri_path, weights_only=True)   # (3, 20, 160, 160)

            # Apply augmentation (train split only)
            if self.augment and self.split == "train":
                mri = augment_mri(mri)

            result["mri"] = mri   # (3, D, H, W)

        return result


# ══════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ══════════════════════════════════════════════════════════

def get_dataloaders(
    config        : DataConfig = None,
    mri_only      : bool = False,
    clinical_only : bool = False,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Args:
        config        : DataConfig (uses defaults if None)
        mri_only      : True for MRI-only model (no clinical features)
        clinical_only : True for PSA-only model (no MRI loading — fast!)

    Returns:
        {
            "train": DataLoader,
            "val":   DataLoader,
            "test":  DataLoader
        }

    Notes:
      - Train loader: shuffle=True, augment=True
      - Val/Test loaders: shuffle=False, augment=False
      - num_workers=0 on Windows (multiprocessing limitation)
    """
    config = config or DataConfig()

    # Detect Windows — multiprocessing doesn't work well on Windows
    import platform
    num_workers = 0 if platform.system() == "Windows" else config.num_workers

    loaders = {}

    for split in ["train", "val", "test"]:
        is_train = (split == "train")

        dataset = PiCAIDataset(
            split         = split,
            config        = config,
            augment       = is_train and config.augment_train,
            mri_only      = mri_only,
            clinical_only = clinical_only,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size  = config.batch_size,
            shuffle     = is_train,       # only shuffle train
            num_workers = num_workers,
            pin_memory  = config.pin_memory and torch.cuda.is_available(),
            drop_last   = is_train,       # drop incomplete last batch in train only
        )

    return loaders


# ══════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════

def sanity_check(config: DataConfig = None):
    """
    Run this to verify the dataset loads correctly before training.

    Checks:
      - All splits load without error
      - Batch shapes are correct
      - Label distribution matches expected ~28% cancer
      - No NaN in MRI or clinical tensors
    """
    config = config or DataConfig()

    print("\n" + "="*60)
    print("DATASET SANITY CHECK")
    print("="*60)

    loaders = get_dataloaders(config)

    for split, loader in loaders.items():
        print(f"\n── {split.upper()} ──")

        batch = next(iter(loader))

        mri      = batch["mri"]        # (B, 3, 20, 160, 160)
        clinical = batch["clinical"]   # (B, 4)
        labels   = batch["label"]      # (B,)
        case_ids = batch["case_id"]    # list of B strings

        print(f"  MRI shape    : {tuple(mri.shape)}")
        print(f"  Clinical     : {tuple(clinical.shape)}")
        print(f"  Labels       : {tuple(labels.shape)}  values={labels.tolist()}")
        print(f"  Case IDs     : {case_ids[:3]}...")
        print(f"  MRI range    : [{mri.min():.3f}, {mri.max():.3f}]")
        print(f"  Clinical     : {clinical[0].tolist()}")
        print(f"  NaN in MRI   : {torch.isnan(mri).any().item()}")
        print(f"  NaN in clin  : {torch.isnan(clinical).any().item()}")

        # Check expected shapes
        assert mri.shape[1:] == (3, 20, 160, 160), f"Wrong MRI shape: {mri.shape}"
        assert clinical.shape[1] == 4,             f"Wrong clinical shape: {clinical.shape}"
        assert not torch.isnan(mri).any(),         "NaN found in MRI!"
        assert not torch.isnan(clinical).any(),    "NaN found in clinical!"
        print(f"  ✓ All checks passed")

    print("\n" + "="*60)
    print("SANITY CHECK COMPLETE — ready to train!")
    print("="*60)


if __name__ == "__main__":
    sanity_check()
