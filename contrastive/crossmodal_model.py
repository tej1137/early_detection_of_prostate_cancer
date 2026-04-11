"""
contrastive/crossmodal_model.py

Cross-Modal Contrastive Representation Model.

Architecture:
    MRI    (3, 20, 160, 160) → MRIEncoder(512) → MRI ProjectionHead    → z_mri    (128, L2-norm)
    Clinical          (4,)   → PSAEncoder(128) → Clinical ProjectionHead→ z_clin   (128, L2-norm)

Loss: Symmetric InfoNCE (identical to CLIP's contrastive objective)
    For a batch of N patients:
        Positive pairs  : (z_mri_i, z_clin_i)   — same patient     ✓
        Negative pairs  : (z_mri_i, z_clin_j)   — different patient ✗
                          (z_clin_i, z_mri_j)   — different patient ✗

    Loss = 0.5 × [InfoNCE(MRI→Clinical) + InfoNCE(Clinical→MRI)]

    This is exactly CLIP's loss but with MRI replacing images and
    clinical features replacing text tokens.

Why this is better than your current approach:
    Current:  SimCLR pretrains MRI alone. PSA encoder is always random.
              The two modalities never interact during pretraining.

    This:     MRI and clinical encoders are trained TOGETHER.
              A patient's MRI embedding and clinical embedding are pulled
              to the same point in the shared 128-dim space.
              After pretraining, both encoders understand "the same patient".

Transfer:
    After pretraining, DISCARD both projection heads.
    Load MRIEncoder(512) + PSAEncoder(128) weights into fusion model.
    Fine-tune with cross-entropy for classification.

Usage:
    from contrastive.crossmodal_model import CrossModalModel, InfoNCELoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mri_baseline.models.mri_encoder import MRIEncoder
from mri_baseline.models.psa_encoder import PSAEncoder


# ══════════════════════════════════════════════════════════
# PROJECTION HEADS
# ══════════════════════════════════════════════════════════

class MRIProjectionHead(nn.Module):
    """
    Projects MRI encoder output → shared embedding space.

    512 → 256 → 128 (L2 normalised)

    Identical structure to SimCLR projection head.
    Discarded after pretraining — only encoder weights transfer.
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
        return F.normalize(self.net(x), dim=1)   # (B, 128) unit sphere


class ClinicalProjectionHead(nn.Module):
    """
    Projects clinical encoder output → shared embedding space.

    128 → 128 → 128 (L2 normalised)

    Smaller than MRI head because PSA encoder output is already compact.
    Discarded after pretraining — only encoder weights transfer.
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)   # (B, 128) unit sphere


# ══════════════════════════════════════════════════════════
# CROSS-MODAL MODEL
# ══════════════════════════════════════════════════════════

class CrossModalModel(nn.Module):
    """
    Dual-encoder cross-modal contrastive model (CLIP-style).

    MRI encoder + Clinical encoder are trained jointly so that
    the same patient's embeddings are close in shared latent space.

    Args:
        mri_encoder_dim    : output dim of MRIEncoder (default 512)
        clinical_encoder_dim: output dim of PSAEncoder (default 128)
        proj_hidden_dim    : projection head hidden dim (default 256 for MRI, 128 for clinical)
        proj_out_dim       : shared embedding dim (default 128)
        temperature        : InfoNCE temperature (default 0.07)
    """

    def __init__(
        self,
        mri_encoder_dim     : int   = 512,
        clinical_encoder_dim: int   = 128,
        proj_out_dim        : int   = 128,
        temperature         : float = 0.07,
    ):
        super().__init__()

        # ── Encoders ───────────────────────────────────────
        self.mri_encoder      = MRIEncoder(embedding_dim=mri_encoder_dim)
        self.clinical_encoder = PSAEncoder(
            in_features   = 4,
            embedding_dim = clinical_encoder_dim,
        )

        # ── Projection heads — discarded after pretraining ─
        self.mri_projector      = MRIProjectionHead(
            in_dim     = mri_encoder_dim,
            hidden_dim = 256,
            out_dim    = proj_out_dim,
        )
        self.clinical_projector = ClinicalProjectionHead(
            in_dim     = clinical_encoder_dim,
            hidden_dim = 128,
            out_dim    = proj_out_dim,
        )

        # ── Learnable temperature (log scale for stability) ─
        # Initialised to match fixed temperature=0.07
        # log(0.07) ≈ -2.66
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Clamped temperature — prevents collapse to 0 or explosion."""
        return self.log_temperature.exp().clamp(min=0.01, max=0.5)

    def forward(
        self,
        mri     : torch.Tensor,
        clinical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both encoders and projection heads.

        Args:
            mri     : (B, 3, 20, 160, 160)
            clinical: (B, 4)

        Returns:
            z_mri   : (B, 128)  L2-normalised MRI embedding
            z_clin  : (B, 128)  L2-normalised clinical embedding
        """
        # Encode
        f_mri  = self.mri_encoder(mri)           # (B, 512)
        f_clin = self.clinical_encoder(clinical)  # (B, 128)

        # Project to shared space
        z_mri  = self.mri_projector(f_mri)       # (B, 128) normalised
        z_clin = self.clinical_projector(f_clin) # (B, 128) normalised

        return z_mri, z_clin

    def encode_mri(self, mri: torch.Tensor) -> torch.Tensor:
        """Raw MRI encoder features — used during fine-tuning."""
        return self.mri_encoder(mri)              # (B, 512)

    def encode_clinical(self, clinical: torch.Tensor) -> torch.Tensor:
        """Raw clinical encoder features — used during fine-tuning."""
        return self.clinical_encoder(clinical)    # (B, 128)

    def get_mri_encoder_weights(self) -> dict:
        """Returns MRI encoder state dict for transfer to fusion model."""
        return self.mri_encoder.state_dict()

    def get_clinical_encoder_weights(self) -> dict:
        """Returns clinical encoder state dict for transfer to fusion model."""
        return self.clinical_encoder.state_dict()


# ══════════════════════════════════════════════════════════
# INFONCE LOSS  (symmetric — identical to CLIP loss)
# ══════════════════════════════════════════════════════════

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Loss for cross-modal contrastive pretraining.

    This is exactly CLIP's contrastive loss:
        Given a batch of N (MRI, clinical) pairs:

        MRI→Clinical direction:
            For each MRI_i, find its matching Clinical_i among N candidates.
            Loss = cross_entropy(sim_matrix_row_i, label=i)

        Clinical→MRI direction:
            For each Clinical_i, find its matching MRI_i among N candidates.
            Loss = cross_entropy(sim_matrix_col_i, label=i)

        Total loss = mean of both directions (symmetric)

    Why symmetric?
        We want BOTH encoders to learn the alignment, not just one.
        MRI encoder learns to predict "which clinical profile matches?"
        Clinical encoder learns to predict "which MRI matches?"

    The temperature controls sharpness:
        Low temperature  → sharp distribution → harder negatives
        High temperature → soft distribution  → easier, less informative
    """

    def forward(
        self,
        z_mri : torch.Tensor,
        z_clin: torch.Tensor,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_mri  : (B, 128) L2-normalised MRI embeddings
            z_clin : (B, 128) L2-normalised clinical embeddings
            temperature: scalar tensor (learnable)

        Returns:
            loss       : scalar — symmetric InfoNCE loss
            loss_mri   : scalar — MRI→clinical direction
            loss_clin  : scalar — clinical→MRI direction
        """
        B      = z_mri.shape[0]
        device = z_mri.device

        # Cosine similarity matrix: (B, B)
        # sim[i,j] = similarity between MRI_i and Clinical_j
        # Since both are L2-normalised: sim = dot product
        sim = torch.mm(z_mri, z_clin.T) / temperature   # (B, B)

        # Diagonal entries are positive pairs (same patient)
        labels = torch.arange(B, device=device)

        # MRI → Clinical: for each row, label = same-column index
        loss_mri  = F.cross_entropy(sim,   labels)

        # Clinical → MRI: for each column, label = same-row index
        loss_clin = F.cross_entropy(sim.T, labels)

        # Symmetric total loss
        loss = (loss_mri + loss_clin) / 2.0

        return loss, loss_mri, loss_clin