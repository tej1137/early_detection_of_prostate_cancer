"""
contrastive/contrastive_dataset.py

Dataset for SimCLR contrastive pretraining.

Returns two independently augmented views of the same MRI scan.
No labels, no clinical features — purely self-supervised.

Usage:
    from contrastive.contrastive_dataset import ContrastiveDataset, get_contrastive_loader
    loader = get_contrastive_loader()
"""

import torch
import torchio as tio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# ══════════════════════════════════════════════════════════
# AUGMENTATION — stronger than supervised, applied twice
# ══════════════════════════════════════════════════════════

def get_contrastive_transform():
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        tio.RandomAffine(scales=0.1, degrees=15, translation=8, p=0.5),
        tio.RandomNoise(std=(0, 0.08), p=0.4),
        tio.RandomBiasField(coefficients=0.4, p=0.4),
        tio.RandomGamma(log_gamma=(-0.4, 0.4), p=0.4),
        tio.RandomBlur(std=(0, 0.5), p=0.2),
    ])


# ══════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════

class ContrastiveDataset(Dataset):
    """
    Loads MRI scans and returns two augmented views per scan.
    Labels are completely ignored — self-supervised only.

    Args:
        split  : "train", "val", or "test" — uses ALL splits by default
        mri_dir: Path to preprocessed .pt MRI files
    """

    def __init__(
        self,
        splits  : list = ["train", "val", "test"],
        mri_dir : Path = Path("/workspace/data/preprocessed/mri"),
        csv_path: Path = Path("/workspace/data/preprocessed/clinical_preprocessed.csv"),
    ):
        self.mri_dir   = mri_dir
        self.transform = get_contrastive_transform()

        # Load all case IDs from specified splits — labels ignored
        df = pd.read_csv(csv_path, index_col="case_id")
        self.case_ids = df[df["split"].isin(splits)].index.tolist()

        print(f"  ContrastiveDataset: {len(self.case_ids)} scans "
              f"(splits: {splits}, no labels used)")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id  = self.case_ids[idx]
        mri_path = self.mri_dir / f"{case_id}.pt"

        mri = torch.load(mri_path, weights_only=True)   # (3, 20, 160, 160)

        # Apply transform twice independently → two different views
        view1 = self._apply_transform(mri)
        view2 = self._apply_transform(mri)

        return view1, view2   # both (3, 20, 160, 160), no labels

    def _apply_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        # torchio expects (C, W, H, D) — permute in and back out
        t = tensor.permute(0, 3, 2, 1)
        subject = tio.Subject(mri=tio.ScalarImage(tensor=t))
        result  = self.transform(subject).mri.tensor
        return result.permute(0, 3, 2, 1).float().clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ══════════════════════════════════════════════════════════

def get_contrastive_loader(
    batch_size  : int  = 16,
    num_workers : int  = 4,
    splits      : list = ["train", "val", "test"],
) -> DataLoader:
    """
    Returns a single DataLoader for contrastive pretraining.
    Uses ALL splits (train+val+test) since no labels are used.
    """
    dataset = ContrastiveDataset(splits=splits)

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,   # NTXent needs full batches
    )
