
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mri_baseline.models.mri_encoder import MRIEncoder
from mri_baseline.models.psa_encoder import PSAEncoder


#DEEPER PROJECTION HEADS

class MRIProjectionHead(nn.Module):
    """
    3-layer MLP: 512 → 256 → 128 → 128 (L2 normalised)

    Extra layer vs crossmodal_model.py gives the MRI encoder
    more capacity to map 3D volumetric features into shared space.

    BatchNorm after each hidden layer stabilises training - critical
    when the two modalities have very different feature scales.
    """

    def __init__(
        self,
        in_dim     : int = 512,
        hidden_dim : int = 256,
        out_dim    : int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)   # (B, 128) unit sphere


class ClinicalProjectionHead(nn.Module):
    """
    3-layer MLP: 256 → 256 → 128 → 128 (L2 normalised)

    Wider than crossmodal_model.py (256 hidden vs 128) because
    clinical encoder is now wider too (256-dim output vs 128-dim).
    """

    def __init__(
        self,
        in_dim     : int = 256,
        hidden_dim : int = 256,
        out_dim    : int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)   # (B, 128) unit sphere


#CLIP MODEL - Option A (from scratch)

class CLIPModel(nn.Module):
    """
    CLIP-style dual encoder — Option A, pure from scratch.

    MRI encoder:      MRIEncoder(512)  — randomly initialised
    Clinical encoder: PSAEncoder(256)  — randomly initialised (wider than crossmodal)

    Both encoders trained jointly via symmetric InfoNCE (CLIP loss).
    Projection heads discarded after pretraining.

    Args:
        mri_encoder_dim     : MRI encoder output dim (default 512)
        clinical_encoder_dim: PSA encoder output dim (default 256, wider than crossmodal)
        proj_out_dim        : shared embedding space dim (default 128)
        init_temperature    : starting temperature (default 0.07, same as CLIP paper)
    """

    def __init__(
        self,
        mri_encoder_dim     : int   = 512,
        clinical_encoder_dim: int   = 256,   #wider than crossmodal_model (128)
        proj_out_dim        : int   = 128,
        init_temperature    : float = 0.07,
    ):
        super().__init__()

        self.mri_encoder_dim      = mri_encoder_dim
        self.clinical_encoder_dim = clinical_encoder_dim

        #Encoders — both randomly initialised
        self.mri_encoder = MRIEncoder(
            in_channels   = 3,
            embedding_dim = mri_encoder_dim,
            dropout       = 0.3,
        )
        self.clinical_encoder = PSAEncoder(
            in_features   = 4,
            embedding_dim = clinical_encoder_dim,
            dropout       = 0.2,
        )

        #Projection heads discarded after pretraining 
        self.mri_projector = MRIProjectionHead(
            in_dim     = mri_encoder_dim,
            hidden_dim = 256,
            out_dim    = proj_out_dim,
        )
        self.clinical_projector = ClinicalProjectionHead(
            in_dim     = clinical_encoder_dim,
            hidden_dim = 256,
            out_dim    = proj_out_dim,
        )

        #  Learnable log-temperature 
        # log(0.07) ≈ -2.659
        # Tighter clamp than crossmodal_model: [0.01, 0.3]
        # CLIP paper found temperature saturates around 0.07–0.1
        self.log_temperature = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Tighter clamp than crossmodal_model — prevents collapse."""
        return self.log_temperature.exp().clamp(min=0.01, max=0.3)

    def forward(
        self,
        mri     : torch.Tensor,
        clinical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mri     : (B, 3, 20, 160, 160)
            clinical: (B, 4)

        Returns:
            z_mri  : (B, 128) L2-normalised MRI embedding
            z_clin : (B, 128) L2-normalised clinical embedding
        """
        f_mri  = self.mri_encoder(mri)            # (B, 512)
        f_clin = self.clinical_encoder(clinical)   # (B, 256)

        z_mri  = self.mri_projector(f_mri)        # (B, 128) normalised
        z_clin = self.clinical_projector(f_clin)  # (B, 128) normalised

        return z_mri, z_clin

    def encode_mri(self, mri: torch.Tensor) -> torch.Tensor:
        """Raw MRI encoder output — used during fine-tuning."""
        return self.mri_encoder(mri)               # (B, 512)

    def encode_clinical(self, clinical: torch.Tensor) -> torch.Tensor:
        """Raw clinical encoder output — used during fine-tuning."""
        return self.clinical_encoder(clinical)     # (B, 256)

    def get_mri_encoder_weights(self) -> dict:
        return self.mri_encoder.state_dict()

    def get_clinical_encoder_weights(self) -> dict:
        return self.clinical_encoder.state_dict()



#CLIP loss compute


class CLIPLoss(nn.Module):
    """
    Symmetric InfoNCE Loss

    For a batch of N (MRI, clinical) pairs:

        Similarity matrix S ∈ R^(N×N):
            S[i,j] = dot(z_mri_i, z_clin_j) / temperature

        MRI → Clinical loss:
            For row i: match z_mri_i to z_clin_i (diagonal is positive)
            loss_mri = cross_entropy(S, labels=[0,1,...,N-1])

        Clinical → MRI loss:
            For col j: match z_clin_j to z_mri_j
            loss_clin = cross_entropy(S.T, labels=[0,1,...,N-1])

        Total = (loss_mri + loss_clin) / 2

    Additional monitoring:
        - Returns diagonal similarity (positive pairs) for logging
        - Returns off-diagonal mean (negative pairs) for logging
    """

    def forward(
        self,
        z_mri      : torch.Tensor,
        z_clin     : torch.Tensor,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_mri      : (B, 128) L2-normalised
            z_clin     : (B, 128) L2-normalised
            temperature: scalar tensor (learnable)

        Returns:
            loss          : scalar — symmetric CLIP loss
            loss_mri      : scalar — MRI→clinical direction
            loss_clin     : scalar — clinical→MRI direction
            pos_sim_mean  : scalar — mean similarity of positive pairs (diagonal)
            neg_sim_mean  : scalar — mean similarity of negative pairs (off-diagonal)
        """
        B      = z_mri.shape[0]
        device = z_mri.device

        # Similarity matrix — (B, B)
        # Both inputs are L2-normalised so this is cosine similarity
        sim    = torch.mm(z_mri, z_clin.T) / temperature

        #Positive pairs are on the diagonal
        labels = torch.arange(B, device=device)

        loss_mri  = F.cross_entropy(sim,   labels)
        loss_clin = F.cross_entropy(sim.T, labels)
        loss      = (loss_mri + loss_clin) / 2.0

        #   Monitoring metrics 
        # Scale sim back to raw cosine similarity for interpretability
        with torch.no_grad():
            raw_sim     = sim * temperature.detach()
            pos_mask    = torch.eye(B, dtype=torch.bool, device=device)
            pos_sim     = raw_sim[pos_mask].mean()
            neg_sim     = raw_sim[~pos_mask].mean()

        return loss, loss_mri, loss_clin, pos_sim, neg_sim



# CLIP MODEL - B SimCLR-warm-started MRI encoder


class CLIPModelB(CLIPModel):
    """
    CLIP-style dual encoder — Option B.

    Identical to CLIPModel (Option A) EXCEPT:
        MRI encoder initialised from SimCLR pretrained best_encoder.pt
        Clinical encoder still randomly initialised

    Args:
        simclr_ckpt: path to best_encoder.pt from train_contrastive.py
        freeze_mri  : if True, MRI encoder frozen during CLIP pretraining
                      (faster, tests whether clinical encoder alone can align)
                      if False, both encoders adapt (recommended)
        + all CLIPModel args
    """

    def __init__(
        self,
        simclr_ckpt         : Path  = Path("/workspace/checkpoints/contrastive/best_encoder.pt"),
        freeze_mri          : bool  = False,
        mri_encoder_dim     : int   = 512,
        clinical_encoder_dim: int   = 256,
        proj_out_dim        : int   = 128,
        init_temperature    : float = 0.07,
    ):
        #Initialise Option A
        super().__init__(
            mri_encoder_dim      = mri_encoder_dim,
            clinical_encoder_dim = clinical_encoder_dim,
            proj_out_dim         = proj_out_dim,
            init_temperature     = init_temperature,
        )

        #Load SimCLR pretrained MRI encoder 
        if not simclr_ckpt.exists():
            raise FileNotFoundError(
                f"SimCLR checkpoint not found: {simclr_ckpt}\n"
                f"Run train_contrastive.py first."
            )

        state = torch.load(simclr_ckpt, map_location="cpu", weights_only=True)
        self.mri_encoder.load_state_dict(state)
        print(f"[Option B] SimCLR MRI encoder loaded from {simclr_ckpt}")

        #  Optionally freeze MRI encoder ─
        if freeze_mri:
            for param in self.mri_encoder.parameters():
                param.requires_grad = False
            print(f" [Option B] MRI encoder FROZEN — only clinical encoder + heads train")
        else:
            print(f" [Option B] MRI encoder UNFROZEN — full CLIP alignment")