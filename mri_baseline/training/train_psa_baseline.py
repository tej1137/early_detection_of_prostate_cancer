"""
training/train_psa_baseline.py

Training script for PSA-only baseline model (Week 3).

This establishes how well clinical features alone can predict cancer.
You already have PSA-only results from your old psa_scripts (0.67-0.72),
but this is the deep learning version for fair comparison with MRI.

Expected performance: 0.68-0.73 AUROC
(Should match or slightly improve your existing PSA baseline)

Purpose: Establish PSA-only baseline before fusion
- MRI-only: 0.72 AUROC (Week 2)
- PSA-only: 0.68-0.73 AUROC (Week 3a)
- Fusion: 0.77-0.82 AUROC (Week 3b)

Usage:
    python -m mri_baseline.training.train_psa_baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json
from datetime import datetime

from mri_baseline.models.psa_encoder import PSAClassifier
from mri_baseline.data.multimodal_dataset import build_dataloaders


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════
class Config:
    """Training hyperparameters"""
    
    # Model
    in_features = 4         # PSA, PSAD, volume, age
    embedding_dim = 256
    num_classes = 2
    dropout = 0.2           # Less dropout than MRI (smaller network)
    
    # Training
    batch_size = 16         # Larger batch (no images, less memory)
    num_epochs = 100
    learning_rate = 1e-3    # Higher LR (simpler model)
    weight_decay = 1e-4
    
    # Early stopping
    patience = 20           # More patience (faster epochs)
    min_delta = 1e-4
    
    # Data
    num_workers = 2
    target_shape = (20, 256, 256)  # Not used, but needed by dataloader
    
    # Paths
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in pbar:
        # Only use clinical features (ignore MRI)
        clinical = batch['clinical'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(clinical)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * clinical.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return running_loss / total, 100. * correct / total


def validate(model, dataloader, criterion, device, epoch):
    """Validate model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            clinical = batch['clinical'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(clinical)
            loss = criterion(logits, labels)
            
            # Probabilities
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # Statistics
            running_loss += loss.item() * clinical.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = running_loss / total
    avg_acc = 100. * correct / total
    auroc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, avg_acc, auroc, all_labels, all_probs


def plot_training_curves(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUROC
    axes[2].plot(epochs, history['val_auroc'], 'g-', linewidth=2)
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUROC')
    axes[2].set_title('Validation AUROC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(labels, probs, threshold=0.5, save_path=None):
    """Plot confusion matrix."""
    predictions = (np.array(probs) >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    classes = ['Benign', 'Cancer']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=20, fontweight='bold')
    
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title('Confusion Matrix - PSA Only', fontsize=16)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════
# MAIN TRAINING
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PSA-ONLY BASELINE TRAINING")
    print("=" * 70)
    
    config = Config()
    
    print(f"\nDevice: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "─" * 70)
    print("LOADING DATA")
    print("─" * 70)
    
    train_loader, val_loader, test_loader, norm_stats = build_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        target_shape=config.target_shape
    )
    
    print(f"\n✓ Data loaded")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Create model
    print("\n" + "─" * 70)
    print("CREATING MODEL")
    print("─" * 70)
    
    model = PSAClassifier(
        in_features=config.in_features,
        embedding_dim=config.embedding_dim,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ PSA classifier created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  (Much smaller than MRI model: ~3.7M)")
    
    # Setup training
    print("\n" + "─" * 70)
    print("TRAINING SETUP")
    print("─" * 70)
    
    class_weights = torch.tensor([1.0, 2.5]).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    print(f"\n✓ Training configured")
    print(f"  Optimizer: Adam (lr={config.learning_rate})")
    print(f"  Note: Higher LR than MRI (simpler model trains faster)")
    
    # Training loop
    print("\n" + "─" * 70)
    print("STARTING TRAINING")
    print("─" * 70)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auroc': []
    }
    
    best_val_loss = float('inf')
    best_auroc = 0.0
    epochs_without_improvement = 0
    best_epoch = 0
    
    start_time = datetime.now()
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{config.num_epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_auroc, val_labels, val_probs = validate(
            model, val_loader, criterion, config.device, epoch
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc)
        
        # Print summary
        print(f"\n{'─'*70}")
        print(f"EPOCH {epoch+1} SUMMARY:")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")
        print(f"  Val AUROC:  {val_auroc:.4f}")
        print(f"{'─'*70}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_auroc = val_auroc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auroc': val_auroc,
                'config': vars(config)
            }
            
            torch.save(checkpoint, config.checkpoint_dir / "best_psa_model.pth")
            
            print(f"\n✓ NEW BEST MODEL SAVED (epoch {epoch+1})")
            print(f"  Val AUROC: {val_auroc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"\n  No improvement for {epochs_without_improvement} epochs")
        
        # Early stopping
        if epochs_without_improvement >= config.patience:
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING at epoch {epoch+1}")
            print(f"Best model was at epoch {best_epoch}")
            print(f"{'='*70}")
            break
    
    training_time = datetime.now() - start_time
    
    # Save results
    print("\n" + "─" * 70)
    print("SAVING RESULTS")
    print("─" * 70)
    
    plot_training_curves(history, config.results_dir / "psa_training_curves.png")
    
    # Load best model
    checkpoint = torch.load(config.checkpoint_dir / "best_psa_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    val_loss, val_acc, val_auroc, val_labels, val_probs = validate(
        model, val_loader, criterion, config.device, epoch=best_epoch-1
    )
    
    plot_confusion_matrix(
        val_labels, val_probs, threshold=0.5,
        save_path=config.results_dir / "psa_confusion_matrix.png"
    )
    
    # Classification report
    val_preds = (np.array(val_probs) >= 0.5).astype(int)
    report = classification_report(
        val_labels, val_preds,
        target_names=['Benign', 'Cancer'],
        digits=4
    )
    
    # Save results
    results = {
        'training_time': str(training_time),
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'best_val_loss': float(best_val_loss),
        'best_val_auroc': float(best_auroc),
        'final_val_auroc': float(val_auroc)
    }
    
    with open(config.results_dir / "psa_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTraining Time: {training_time}")
    print(f"Best Epoch: {best_epoch}")
    print(f"\nPSA-Only Results:")
    print(f"  Validation AUROC: {val_auroc:.4f}")
    print(f"\nClassification Report:")
    print(report)
    
    # Target check
    if val_auroc >= 0.68:
        print("\n✓ SUCCESS: PSA baseline target (0.68) achieved!")
    else:
        print(f"\n⚠ Below target: {val_auroc:.4f} < 0.68")
    
    print("\nFiles saved:")
    print(f"  Model:      {config.checkpoint_dir / 'best_psa_model.pth'}")
    print(f"  Curves:     {config.results_dir / 'psa_training_curves.png'}")
    print(f"  Confusion:  {config.results_dir / 'psa_confusion_matrix.png'}")
    print(f"  Results:    {config.results_dir / 'psa_results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
