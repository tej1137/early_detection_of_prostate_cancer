"""
models/mri_encoder.py

3D CNN Encoder for full MRI volumes (now that we have GPU).

Input:  [B, 3, 20, 256, 256]  — batch of MRI volumes, 3 channels
Output: [B, 512]               — batch of embedding vectors

Architecture based on:
- 3D U-Net style encoder (Çiçek et al., MICCAI 2016)
- Batch normalisation (Ioffe & Szegedy, 2015)
- Global Average Pooling (Lin et al., 2013)

The 3D approach captures inter-slice relationships which 2.5D misses.
For example, a lesion that appears across multiple slices is easier
to identify in 3D than looking at slices independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """
    One 3D convolutional block:
        Conv3D → BatchNorm → ReLU → Conv3D → BatchNorm → ReLU → MaxPool3D

    Args:
        in_channels  : number of input feature maps
        out_channels : number of output feature maps
        pool         : if True, apply MaxPool3d to halve spatial dimensions
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()

        layers = [
            # First 3D conv
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            # Second 3D conv
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if pool:
            # Halve all spatial dimensions: (20, 256, 256) → (10, 128, 128)
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MRIEncoder(nn.Module):
    """
    3D CNN Encoder: MRI volume → embedding vector

    Architecture:
        Input  [B, 3, 20, 256, 256]
          ↓ Conv3DBlock(3 → 32)    + pool  →  [B, 32,  10, 128, 128]
          ↓ Conv3DBlock(32 → 64)   + pool  →  [B, 64,  5,  64,  64 ]
          ↓ Conv3DBlock(64 → 128)  + pool  →  [B, 128, 2,  32,  32 ]
          ↓ Conv3DBlock(128 → 256) + pool  →  [B, 256, 1,  16,  16 ]
          ↓ GlobalAvgPool3D                →  [B, 256]
          ↓ Dropout(0.3)
          ↓ Linear(256 → 512)              →  [B, 512]
        Output [B, 512]

    Args:
        in_channels   : number of MRI channels (3 = T2W, ADC, HBV)
        embedding_dim : size of output embedding vector (default 512)
        dropout       : dropout rate (default 0.3)
    """
    def __init__(self,
                 in_channels:   int = 3,
                 embedding_dim: int = 512,
                 dropout:       float = 0.3):
        super().__init__()

        # ── 3D Convolutional backbone ──────────────────────
        self.conv_blocks = nn.Sequential(
            Conv3DBlock(in_channels, 32,  pool=True),   # 20×256×256 → 10×128×128
            Conv3DBlock(32,          64,  pool=True),   # 10×128×128 → 5×64×64
            Conv3DBlock(64,          128, pool=True),   # 5×64×64    → 2×32×32
            Conv3DBlock(128,         256, pool=True),   # 2×32×32    → 1×16×16
        )

        # ── Global Average Pooling ─────────────────────────
        # Takes [B, 256, 1, 16, 16] → [B, 256]
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # ── Projection head ────────────────────────────────
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        )

        # ── Weight initialisation ──────────────────────────
        self._initialise_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, 20, 256, 256] — batch of MRI volumes

        Returns:
            embedding : [B, 512] — feature vectors
        """
        features = self.conv_blocks(x)          # [B, 256, 1, 16, 16]
        pooled   = self.global_avg_pool(features)  # [B, 256, 1, 1, 1]
        pooled   = pooled.flatten(start_dim=1)  # [B, 256]
        embedding = self.projection(pooled)     # [B, 512]
        return embedding

    def _initialise_weights(self):
        """Kaiming initialisation for 3D conv layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias,   0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)


class MRIClassifier(nn.Module):
    """
    Full MRI-only classifier for the BASELINE experiment.
    Encoder + classification head.

    Input:  [B, 3, 20, 256, 256]
    Output: [B, 2]  — logits for [benign, cancer]
    """
    def __init__(self,
                 in_channels:   int = 3,
                 embedding_dim: int = 512,
                 num_classes:   int = 2,
                 dropout:       float = 0.3):
        super().__init__()

        self.encoder = MRIEncoder(in_channels, embedding_dim, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, 20, 256, 256]

        Returns:
            logits : [B, 2]
        """
        embedding = self.encoder(x)
        logits    = self.classifier(embedding)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns cancer probability for AUROC calculation.

        Returns:
            probs : [B] — P(cancer) for each sample
        """
        logits = self.forward(x)
        probs  = F.softmax(logits, dim=1)
        return probs[:, 1]   # cancer class


# ─────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("3D MRI Encoder — Sanity Check")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Test MRIEncoder ────────────────────────────────────
    print("\n[1] Testing MRIEncoder...")
    encoder = MRIEncoder(in_channels=3, embedding_dim=512).to(device)

    dummy_input = torch.randn(2, 3, 20, 256, 256).to(device)
    embedding   = encoder(dummy_input)

    print(f"  Input shape     : {dummy_input.shape}")
    print(f"  Embedding shape : {embedding.shape}")
    assert embedding.shape == (2, 512), "Wrong embedding shape!"
    print("  ✓ PASSED")

    # ── Test MRIClassifier ─────────────────────────────────
    print("\n[2] Testing MRIClassifier...")
    classifier = MRIClassifier(in_channels=3, embedding_dim=512).to(device)

    logits = classifier(dummy_input)
    probs  = classifier.predict_proba(dummy_input)

    print(f"  Logits shape : {logits.shape}")
    print(f"  Proba shape  : {probs.shape}")
    print(f"  Probabilities: {probs.detach().cpu().numpy()}")
    assert logits.shape == (2, 2), "Wrong logits shape!"
    assert probs.shape  == (2,),   "Wrong proba shape!"
    print("  ✓ PASSED")

    # ── Memory usage ───────────────────────────────────────
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        mem_reserved  = torch.cuda.memory_reserved(0) / 1e9
        print(f"\nGPU Memory:")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved:  {mem_reserved:.2f} GB")

    # ── Parameter count ────────────────────────────────────
    total_params = sum(p.numel() for p in classifier.parameters())
    train_params = sum(p.numel() for p in classifier.parameters()
                       if p.requires_grad)
    print(f"\nModel Size:")
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {train_params:,}")

    print("\n✓ SANITY CHECK PASSED — 3D model ready for training")