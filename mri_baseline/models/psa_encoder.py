"""
models/psa_encoder.py

MLP Encoder for clinical features (PSA, PSAD, prostate volume, age).
Converts 4 raw clinical values into a 256-dimensional learned representation.

This pairs with MRIEncoder to create a multimodal system:
  MRI:  [B, 3, 20, 256, 256] → MRIEncoder  → [B, 512]
  PSA:  [B, 4]               → PSAEncoder  → [B, 256]
  Fused: [B, 768] → Classifier → [B, 2]

Usage:
    from mri_baseline.models.psa_encoder import PSAEncoder, PSAClassifier
    
    encoder = PSAEncoder(in_features=4, embedding_dim=256)
    embedding = encoder(clinical_features)  # [B, 4] → [B, 256]
"""

import torch
import torch.nn as nn


class PSAEncoder(nn.Module):
    """
    MLP Encoder for clinical features.
    
    Architecture:
        Input: [B, 4]  (PSA, PSAD, prostate_volume, patient_age)
          ↓
        Linear(4 → 64) + ReLU + Dropout(0.2)
          ↓
        Linear(64 → 128) + ReLU + Dropout(0.2)
          ↓
        Linear(128 → 256) + ReLU
          ↓
        Output: [B, 256]  (clinical embedding)
    
    Why this architecture?
    - Progressively expands: 4 → 64 → 128 → 256
    - Learns non-linear combinations of features
    - Dropout prevents overfitting on small feature set
    - Matches dimension ratio with MRI encoder (512:256 = 2:1)
    
    Args:
        in_features: Number of input features (default 4)
        embedding_dim: Size of output embedding (default 256)
        dropout: Dropout rate (default 0.2)
    """
    
    def __init__(self, 
                 in_features: int = 4,
                 embedding_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1: 4 → 64
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 2: 64 → 128
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 3: 128 → 256
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._initialise_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, 4] clinical features (already normalized)
        
        Returns:
            embedding: [B, 256] learned representation
        """
        return self.encoder(x)
    
    def _initialise_weights(self):
        """
        Xavier initialization for linear layers.
        Good default for layers with ReLU activation.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)


class PSAClassifier(nn.Module):
    """
    Complete PSA-only classifier (for comparison baseline).
    
    This is used to establish PSA-only performance before fusion.
    You already have PSA-only results (0.67-0.72 AUROC) from your
    old psa_scripts, but this is the deep learning version.
    
    Architecture:
        PSAEncoder: [4] → [256]
        Classifier: [256] → [128] → [2]
    
    Expected performance: ~0.68-0.73 AUROC
    (Should match or slightly improve your existing PSA baseline)
    """
    
    def __init__(self,
                 in_features: int = 4,
                 embedding_dim: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.encoder = PSAEncoder(in_features, embedding_dim, dropout)
        
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
            x: [B, 4] clinical features
        
        Returns:
            logits: [B, 2] class scores
        """
        embedding = self.encoder(x)      # [B, 256]
        logits = self.classifier(embedding)  # [B, 2]
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns cancer probability for AUROC calculation.
        
        Returns:
            probs: [B] probability of cancer
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # P(cancer)


# ═════════════════════════════════════════════════════════
# Sanity Check
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("PSA Encoder — Sanity Check")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Test PSAEncoder
    print("\n[1] Testing PSAEncoder...")
    encoder = PSAEncoder(in_features=4, embedding_dim=256).to(device)
    
    # Simulate normalized clinical features
    dummy_input = torch.randn(8, 4).to(device)  # batch of 8
    embedding = encoder(dummy_input)
    
    print(f"  Input shape  : {dummy_input.shape}")  # [8, 4]
    print(f"  Output shape : {embedding.shape}")     # [8, 256]
    assert embedding.shape == (8, 256), "Wrong shape!"
    print("  ✓ PASSED")
    
    # Test PSAClassifier
    print("\n[2] Testing PSAClassifier...")
    classifier = PSAClassifier(in_features=4, embedding_dim=256).to(device)
    
    logits = classifier(dummy_input)
    probs = classifier.predict_proba(dummy_input)
    
    print(f"  Logits shape : {logits.shape}")   # [8, 2]
    print(f"  Probs shape  : {probs.shape}")    # [8]
    print(f"  Probabilities: {probs.detach().cpu().numpy()}")
    assert logits.shape == (8, 2)
    assert probs.shape == (8,)
    print("  ✓ PASSED")
    
    # Parameter count
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✓ SANITY CHECK PASSED — PSA encoder ready")
