"""
models/fusion_model.py

Multimodal Fusion Model - combines MRI and PSA encoders.

This is Week 3's main model. It takes both imaging and clinical data,
encodes each separately, then fuses them for classification.

Architecture:
    MRI Branch:  [B, 3, 20, 256, 256] → MRIEncoder  → [B, 512]
    PSA Branch:  [B, 4]               → PSAEncoder  → [B, 256]
    Fusion:      [B, 768] (concatenate)
    Classifier:  [B, 768] → [B, 256] → [B, 2]

Expected performance: 0.77-0.82 AUROC
(+5-10% improvement over MRI-only baseline of 0.72)

Usage:
    from mri_baseline.models.fusion_model import MultimodalFusionModel
    
    model = MultimodalFusionModel()
    logits = model(mri_images, clinical_features)
"""

import torch
import torch.nn as nn

from mri_baseline.models.mri_encoder import MRIEncoder
from mri_baseline.models.psa_encoder import PSAEncoder


class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion via concatenation.
    
    Why concatenation?
    - Simple and effective baseline fusion strategy
    - Preserves all information from both modalities
    - Let's the classifier learn how to weight each modality
    - Other strategies (addition, attention) can be tried later
    
    Architecture breakdown:
    1. MRI Encoder:    3D CNN extracts spatial features → 512-dim
    2. PSA Encoder:    MLP extracts clinical features → 256-dim
    3. Concatenation:  Stack both embeddings → 768-dim
    4. Fusion Layer:   Learn cross-modal interactions → 256-dim
    5. Classifier:     Final decision → 2 classes
    
    Args:
        mri_in_channels: Number of MRI channels (default 3: T2W, ADC, HBV)
        mri_embedding_dim: Size of MRI embedding (default 512)
        psa_in_features: Number of clinical features (default 4)
        psa_embedding_dim: Size of PSA embedding (default 256)
        num_classes: Number of output classes (default 2)
        dropout: Dropout rate (default 0.3)
    """
    
    def __init__(self,
                 mri_in_channels: int = 3,
                 mri_embedding_dim: int = 512,
                 psa_in_features: int = 4,
                 psa_embedding_dim: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        # ═══ ENCODERS ═══
        self.mri_encoder = MRIEncoder(
            in_channels=mri_in_channels,
            embedding_dim=mri_embedding_dim,
            dropout=dropout
        )
        
        self.psa_encoder = PSAEncoder(
            in_features=psa_in_features,
            embedding_dim=psa_embedding_dim,
            dropout=dropout * 0.67  # Slightly less dropout for smaller network
        )
        
        # ═══ FUSION LAYER ═══
        # Concatenated dimension
        fusion_dim = mri_embedding_dim + psa_embedding_dim  # 512 + 256 = 768
        
        # Learn cross-modal interactions
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # ═══ CLASSIFIER ═══
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, 
                mri: torch.Tensor,
                clinical: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with both modalities.
        
        Args:
            mri: [B, 3, 20, 256, 256] MRI volumes
            clinical: [B, 4] clinical features (PSA, PSAD, volume, age)
        
        Returns:
            logits: [B, 2] class scores (benign, cancer)
        """
        # Encode each modality separately
        mri_emb = self.mri_encoder(mri)        # [B, 512]
        psa_emb = self.psa_encoder(clinical)   # [B, 256]
        
        # Concatenate embeddings
        fused = torch.cat([mri_emb, psa_emb], dim=1)  # [B, 768]
        
        # Learn cross-modal interactions
        fused = self.fusion_layer(fused)       # [B, 256]
        
        # Final classification
        logits = self.classifier(fused)        # [B, 2]
        
        return logits
    
    def predict_proba(self,
                      mri: torch.Tensor,
                      clinical: torch.Tensor) -> torch.Tensor:
        """
        Returns cancer probability.
        
        Returns:
            probs: [B] P(cancer) for each sample
        """
        logits = self.forward(mri, clinical)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]
    
    def get_embeddings(self,
                       mri: torch.Tensor,
                       clinical: torch.Tensor) -> dict:
        """
        Extract embeddings from each modality.
        Useful for visualization and analysis.
        
        Returns:
            dict with keys: 'mri', 'psa', 'fused'
        """
        mri_emb = self.mri_encoder(mri)
        psa_emb = self.psa_encoder(clinical)
        fused = torch.cat([mri_emb, psa_emb], dim=1)
        fused = self.fusion_layer(fused)
        
        return {
            'mri': mri_emb,       # [B, 512]
            'psa': psa_emb,       # [B, 256]
            'fused': fused        # [B, 256]
        }


# ═════════════════════════════════════════════════════════
# Sanity Check
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("Multimodal Fusion Model — Sanity Check")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model
    print("\n[1] Creating multimodal model...")
    model = MultimodalFusionModel().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\n[2] Testing forward pass...")
    dummy_mri = torch.randn(2, 3, 20, 256, 256).to(device)
    dummy_psa = torch.randn(2, 4).to(device)
    
    print(f"  MRI input shape:      {dummy_mri.shape}")
    print(f"  Clinical input shape: {dummy_psa.shape}")
    
    logits = model(dummy_mri, dummy_psa)
    probs = model.predict_proba(dummy_mri, dummy_psa)
    
    print(f"  Logits shape:         {logits.shape}")    # [2, 2]
    print(f"  Probs shape:          {probs.shape}")     # [2]
    print(f"  Cancer probabilities: {probs.detach().cpu().numpy()}")
    
    assert logits.shape == (2, 2)
    assert probs.shape == (2,)
    print("  ✓ PASSED")
    
    # Test embeddings extraction
    print("\n[3] Testing embedding extraction...")
    embeddings = model.get_embeddings(dummy_mri, dummy_psa)
    
    print(f"  MRI embedding shape:   {embeddings['mri'].shape}")    # [2, 512]
    print(f"  PSA embedding shape:   {embeddings['psa'].shape}")    # [2, 256]
    print(f"  Fused embedding shape: {embeddings['fused'].shape}")  # [2, 256]
    
    assert embeddings['mri'].shape == (2, 512)
    assert embeddings['psa'].shape == (2, 256)
    assert embeddings['fused'].shape == (2, 256)
    print("  ✓ PASSED")
    
    # Memory check
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        mem_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"\nGPU Memory:")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved:  {mem_reserved:.2f} GB")
    
    print("\n✓ SANITY CHECK PASSED — Fusion model ready for training")
