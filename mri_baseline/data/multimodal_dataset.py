"""
data/multimodal_dataset.py

PyTorch Dataset that pairs MRI volumes with clinical features.
This is the single class used by ALL training scripts.

Usage:
    from data.multimodal_dataset import MultimodalDataset, build_dataloaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from mri_baseline.data.load_mri import load_mri_case, get_all_case_ids, IMAGE_DIR
from mri_baseline.data.load_psa import (
    load_clinical_data, fit_normalisation,
    get_clinical_vector, get_label,
    split_case_ids, MARKSHEET_PATH
)


class MultimodalDataset(Dataset):
    """
    Returns one sample per __getitem__:
    {
      "image"    : FloatTensor  (3, Z, H, W)   — MRI (T2W, ADC, HBV)
      "clinical" : FloatTensor  (4,)            — PSA, PSAD, volume, age
      "label"    : LongTensor   scalar          — 0 = benign, 1 = cancer
      "case_id"  : str                          — for debugging
    }
    """

    def __init__(self,
                 case_ids:    list,
                 clinical_df,
                 norm_stats:  dict,
                 image_dir:   Path   = IMAGE_DIR,
                 target_shape: tuple = (20, 256, 256),
                 augment:     bool   = False):
        """
        Args:
            case_ids     : list of "patient_id_study_id" strings
            clinical_df  : DataFrame from load_clinical_data()
            norm_stats   : dict from fit_normalisation()  (fit on TRAIN only)
            image_dir    : root folder of PI-CAI images
            target_shape : (Z, H, W) each volume is resized to this
            augment      : if True, apply random augmentations (train only)
        """
        self.case_ids     = case_ids
        self.clinical_df  = clinical_df
        self.norm_stats   = norm_stats
        self.image_dir    = image_dir
        self.target_shape = target_shape
        self.augment      = augment

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]

        # ── parse patient_id and study_id ──────────────
        # case_id format: "10000_1000000"
        parts      = case_id.split("_")
        patient_id = parts[0]
        study_id   = parts[1]

        # ── MODALITY 1: MRI ────────────────────────────
        try:
            image = load_mri_case(
                patient_id, study_id,
                image_dir=self.image_dir,
                target_shape=self.target_shape
            )
        except FileNotFoundError as e:
            # Graceful fallback: return zeros so training doesn't crash
            print(f"WARNING: {e}")
            z, h, w = self.target_shape
            image   = np.zeros((3, z, h, w), dtype=np.float32)

        if self.augment:
            image = self._augment(image)

        # ── MODALITY 2: CLINICAL ───────────────────────
        clinical = get_clinical_vector(case_id, self.clinical_df, self.norm_stats)

        # ── LABEL ──────────────────────────────────────
        try:
            label = get_label(case_id, self.clinical_df)
        except KeyError:
            label = 0     # default to benign if missing

        return {
            "image":    torch.from_numpy(image).float(),
            "clinical": torch.from_numpy(clinical).float(),
            "label":    torch.tensor(label, dtype=torch.long),
            "case_id":  case_id
        }

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        """
        Simple augmentations that are safe for medical images:
        - Random horizontal flip
        - Random vertical flip
        - Mild random Gaussian noise
        All applied with 50% probability.
        """
        # Flip along H axis
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()

        # Flip along W axis
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=3).copy()

        # Add small Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
            image = image + noise

        return image


def build_dataloaders(image_dir:    Path  = IMAGE_DIR,
                      marksheet:    Path  = MARKSHEET_PATH,
                      batch_size:   int   = 2,
                      num_workers:  int   = 2,
                      target_shape: tuple = (20, 256, 256),
                      seed:         int   = 42):
    """
    One-call function: loads data, splits, normalises, returns DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, norm_stats
    """

    # 1. Load clinical data
    clinical_df = load_clinical_data(marksheet)

    # 2. Train/val/test split
    train_ids, val_ids, test_ids = split_case_ids(clinical_df, seed=seed)

    # 3. Fit normalisation on training set ONLY
    train_df   = clinical_df.loc[train_ids]
    norm_stats = fit_normalisation(train_df)

    # 4. Create datasets
    train_ds = MultimodalDataset(
        train_ids, clinical_df, norm_stats,
        image_dir=image_dir,
        target_shape=target_shape,
        augment=True              # augment training only
    )
    val_ds = MultimodalDataset(
        val_ids, clinical_df, norm_stats,
        image_dir=image_dir,
        target_shape=target_shape,
        augment=False
    )
    test_ds = MultimodalDataset(
        test_ids, clinical_df, norm_stats,
        image_dir=image_dir,
        target_shape=target_shape,
        augment=False
    )

    # 5. Wrap in DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,             # always batch=1 for test
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoaders ready:")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    return train_loader, val_loader, test_loader, norm_stats


# ─────────────────────────────────────────────
# Quick sanity check — run this file directly
# python data/multimodal_dataset.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("Building dataloaders...")
    train_loader, val_loader, test_loader, norm_stats = build_dataloaders(
        batch_size=2,
        target_shape=(20, 256, 256)
    )

    print("\nChecking first training batch...")
    batch = next(iter(train_loader))

    print(f"  image shape   : {batch['image'].shape}")      # [2, 3, 20, 256, 256]
    print(f"  clinical shape: {batch['clinical'].shape}")   # [2, 4]
    print(f"  labels        : {batch['label']}")            # [0 or 1, 0 or 1]
    print(f"  case_ids      : {batch['case_id']}")

    assert batch["image"].shape[1]    == 3,  "Expected 3 MRI channels"
    assert batch["clinical"].shape[1] == 4,  "Expected 4 clinical features"
    assert batch["image"].dtype       == torch.float32
    assert batch["clinical"].dtype    == torch.float32
    assert batch["label"].dtype       == torch.int64

    print("\nAll assertions passed — dataset is working correctly!")
