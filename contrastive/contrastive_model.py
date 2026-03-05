"""
contrastive/contrastive_model.py

SimCLR-style contrastive model for MRI pretraining.

Architecture:
    MRI input (3, 20, 160, 160)
        → 3D CNN Encoder       (reuses MRIEncoder from mri_baseline)
        → Projection Head      (MLP: 512 → 256 → 128)
        → L2-normalised vector (128-dim)

The projection head is only used during pretraining.
It is discarded during fine-tuning — only the encoder weights are kept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mri_baseline.models.mri_encoder import MRIEncoder


# ══════════════════════════════════════════════════════════
# PROJECTION HEAD
# ══════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """
    2-layer MLP that maps encoder output → contrastive embedding space.
    Discarded after pretraining — only encoder weights are transferred.

    512 → 256 → 128 (L2 normalised)
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)   # L2 normalise → unit sphere


# ══════════════════════════════════════════════════════════
# SIMCLR MODEL
# ══════════════════════════════════════════════════════════

class SimCLRModel(nn.Module):
    """
    Full SimCLR model: Encoder + Projection Head.

    Args:
        encoder_out_dim : output dim of MRIEncoder (default 512)
        proj_hidden_dim : projection head hidden dim (default 256)
        proj_out_dim    : final embedding dim (default 128)
        freeze_encoder  : freeze encoder weights during pretraining (default False)
    """

    def __init__(
        self,
        encoder_out_dim : int  = 512,
        proj_hidden_dim : int  = 256,
        proj_out_dim    : int  = 128,
        freeze_encoder  : bool = False,
    ):
        super().__init__()

        self.encoder    = MRIEncoder(embedding_dim=encoder_out_dim)
        self.projector  = ProjectionHead(encoder_out_dim, proj_hidden_dim, proj_out_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised projection embedding."""
        features = self.encoder(x)        # (B, 512)
        return self.projector(features)   # (B, 128) normalised

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw encoder features — used during fine-tuning.
        Bypasses projection head entirely.
        """
        return self.encoder(x)            # (B, 512)

    def get_encoder_weights(self) -> dict:
        """Returns just the encoder state dict for transfer learning."""
        return self.encoder.state_dict()


# ══════════════════════════════════════════════════════════
# NTXENT LOSS (SimCLR Loss)
# ══════════════════════════════════════════════════════════

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss.

    For a batch of N scans with 2 views each (2N total):
    - Positive pair: (view1_i, view2_i) — same scan
    - Negative pairs: all other 2(N-1) views in the batch

    Lower temperature → sharper, harder contrasts.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: embeddings from view 1, shape (B, 128), L2 normalised
            z2: embeddings from view 2, shape (B, 128), L2 normalised
        Returns:
            scalar loss
        """
        B = z1.shape[0]
        device = z1.device

        # Concatenate all embeddings: (2B, 128)
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity on diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pair indices
        # z1[i] pairs with z2[i] → index i pairs with index i+B
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B,     device=device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss
