import torch
import torch.nn as nn
import torch.nn.functional as F

#Conv3DBlock is identical to mri_encoder.py - reused without modification.

class Conv3DBlock(nn.Module):
    """
    3D convolutional block: Conv→BN→ReLU→Conv→BN→ReLU→(MaxPool)
    Identical to mri_encoder.py — reused without modification.
    """

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()

        layers = [
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


#BiMRI Encoder and Classifier

class BiMRIEncoder(nn.Module):
    """
    3D CNN Encoder for bi-parametric MRI.

    Identical to MRIEncoder except in_channels defaults to 2.
    All other hyperparameters unchanged for fair comparison.

    Args:
        in_channels  : number of MRI channels (2 = T2W + ADC)
        embedding_dim: size of output embedding vector (default 512)
        dropout      : dropout rate (default 0.3)
    """

    def __init__(
        self,
        in_channels  : int   = 2,      #2 not 3
        embedding_dim: int   = 512,
        dropout      : float = 0.3,
    ):
        super().__init__()

        #3D Convolutional 
        #Identical channel progression to MRIEncoder
        self.conv_blocks = nn.Sequential(
            Conv3DBlock(in_channels, 32,  pool=True),
            Conv3DBlock(32,          64,  pool=True),
            Conv3DBlock(64,          128, pool=True),
            Conv3DBlock(128,         256, pool=True),
        )

        #GAPPP 
        #AdaptiveAvgPool3d handles any spatial size -- always (B, 256)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        #Projection
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        )

        self._initialise_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 2, 20, 160, 160) — T2W + ADC only

        Returns:
            embedding : (B, 512)
        """
        features  = self.conv_blocks(x)          
        pooled    = self.global_avg_pool(features) 
        pooled    = pooled.flatten(start_dim=1)  
        embedding = self.projection(pooled)    
        return embedding

    def _initialise_weights(self):
        """Kaiming initialisation — identical to MRIEncoder."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias,   0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)



class BiMRIClassifier(nn.Module):
    """
    Full biMRI classifier: BiMRIEncoder + classification head.

    Identical structure to MRIClassifier in mri_encoder.py.
    Only difference: uses BiMRIEncoder (in_channels=2).

    Input:  (B, 2, 20, 160, 160)  ← T2W + ADC
    Output: (B, 2)                ← logits [benign, cancer]
    """

    def __init__(
        self,
        in_channels  : int   = 2,
        embedding_dim: int   = 512,
        num_classes  : int   = 2,
        dropout      : float = 0.3,
    ):
        super().__init__()

        self.encoder = BiMRIEncoder(in_channels, embedding_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 2, 20, 160, 160)
        Returns:
            logits : (B, 2)
        """
        embedding = self.encoder(x)
        return self.classifier(embedding)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(cancer) for AUROC calculation."""
        return F.softmax(self.forward(x), dim=1)[:, 1]


#Check and parameter count

if __name__ == "__main__":
    print("=" * 50)
    print("BiMRI Encoder — Sanity Check")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    #Test BiMRIEncoder
    print("\n[1] Testing BiMRIEncoder (2 channels)")
    encoder = BiMRIEncoder(in_channels=2, embedding_dim=512).to(device)
    dummy   = torch.randn(2, 2, 20, 160, 160).to(device)
    emb     = encoder(dummy)
    print(f"  Input  : {tuple(dummy.shape)}")
    print(f"  Output : {tuple(emb.shape)}")
    assert emb.shape == (2, 512)
    print("  ✓ PASSED")

    #Test BiMRIClassifier
    print("\n[2] Testing BiMRIClassifier...")
    model  = BiMRIClassifier(in_channels=2).to(device)
    logits = model(dummy)
    probs  = model.predict_proba(dummy)
    print(f"  Logits : {tuple(logits.shape)}")
    print(f"  Probs  : {tuple(probs.shape)}")
    assert logits.shape == (2, 2)
    assert probs.shape  == (2,)
    print("PASSED")

    #Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters : {total:,}")
    print(f"  (mpMRI encoder   : ~3,647,392 — biMRI slightly fewer due to fewer input channels)")
    print("\nSANITY CHECK PASSED biMRI model ready")