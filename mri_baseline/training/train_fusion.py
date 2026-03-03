"""
training/train_fusion.py
FIXED: Absolute paths, AUROC checkpointing, WeightedRandomSampler,
       class_weight [1.0, 2.5], verbose removed
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json
from datetime import datetime

from mri_baseline.models.fusion_model import MultimodalFusionModel
from mri_baseline.data.multimodal_dataset import build_dataloaders


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════
class Config:
    mri_in_channels   = 3
    mri_embedding_dim = 512
    psa_in_features   = 4
    psa_embedding_dim = 256
    num_classes       = 2
    dropout           = 0.3

    batch_size    = 4
    num_epochs    = 100
    learning_rate = 1e-4
    weight_decay  = 1e-4

    patience  = 15
    min_delta = 1e-4

    num_workers  = 4
    target_shape = (20, 256, 256)

    project_root   = Path("/workspace/early_detection_of_prostate_cancer")
    checkpoint_dir = project_root / "checkpoints"
    results_dir    = project_root / "results"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# WEIGHTED SAMPLER
# ══════════════════════════════════════════════════════════

def build_weighted_sampler(dataset):
    labels = np.array([
        int(dataset.clinical_df.loc[case_id, "case_csPCa"])
        for case_id in dataset.case_ids
    ])
    class_counts   = np.bincount(labels)
    print(f"  Train class distribution: Benign={class_counts[0]}, Cancer={class_counts[1]}")
    class_weights  = 1.0 / class_counts
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ══════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in pbar:
        images   = batch['image'].to(device)
        clinical = batch['clinical'].to(device)
        labels   = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(images, clinical)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / total, 100. * correct / total


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for batch in pbar:
            images   = batch['image'].to(device)
            clinical = batch['clinical'].to(device)
            labels   = batch['label'].to(device)

            logits = model(images, clinical)
            loss   = criterion(logits, labels)
            probs  = torch.softmax(logits, dim=1)[:, 1]

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    auroc = roc_auc_score(all_labels, all_probs)
    return running_loss / total, 100. * correct / total, auroc, all_labels, all_probs


# ══════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════

def plot_training_curves(history, save_path, mri_baseline=0.72):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'],   'r-', label='Val',   linewidth=2)
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'],   'r-', label='Val',   linewidth=2)
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['val_auroc'], 'g-', linewidth=2, label='Fusion')
    axes[2].axhline(y=0.5,          color='gray',   linestyle='--', alpha=0.5, label='Random')
    axes[2].axhline(y=mri_baseline, color='orange', linestyle='--', alpha=0.7, label=f'MRI baseline ({mri_baseline})')
    axes[2].set_title('Val AUROC', fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUROC')
    axes[2].set_ylim([0.4, 1.0]); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved: {save_path}")
    plt.close()


def plot_confusion_matrix(labels, probs, threshold=0.5, save_path=None):
    predictions = (np.array(probs) >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    classes = ['Benign', 'Cancer']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=20, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('True', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - Fusion Model', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MULTIMODAL FUSION TRAINING (Fixed)")
    print("=" * 70)

    config = Config()
    print(f"\nDevice: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── DATA ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("LOADING DATA + BUILDING SAMPLER")
    print("─" * 70)

    train_loader, val_loader, test_loader, norm_stats = build_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        target_shape=config.target_shape,
    )

    print("Building WeightedRandomSampler...")
    sampler = build_weighted_sampler(train_loader.dataset)
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    print(f"✓ Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── MODEL ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("CREATING MODEL")
    print("─" * 70)

    model = MultimodalFusionModel(
        mri_in_channels=config.mri_in_channels,
        mri_embedding_dim=config.mri_embedding_dim,
        psa_in_features=config.psa_in_features,
        psa_embedding_dim=config.psa_embedding_dim,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Fusion model | Parameters: {total_params:,}")

    # ── TRAINING SETUP ────────────────────────────────────
    print("\n" + "─" * 70)
    print("TRAINING SETUP")
    print("─" * 70)

    class_weights = torch.tensor([1.0, 2.5]).to(config.device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print(f"✓ Loss: CrossEntropyLoss [benign=1.0, cancer=2.5]")
    print(f"✓ Optimizer: Adam (lr={config.learning_rate})")
    print(f"✓ Sampler: WeightedRandomSampler")
    print(f"✓ Checkpointing: AUROC-based")
    print(f"✓ Early stopping patience: {config.patience}")

    # ── TRAINING LOOP ─────────────────────────────────────
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auroc': []}
    best_auroc    = 0.0
    best_val_loss = float('inf')
    best_epoch    = 0
    epochs_without_improvement = 0
    start_time = datetime.now()

    for epoch in range(config.num_epochs):
        print(f"\n{'='*70}\nEPOCH {epoch+1}/{config.num_epochs}\n{'='*70}")

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.device, epoch)
        v_loss, v_acc, v_auroc, v_labels, v_probs = validate(model, val_loader, criterion, config.device, epoch)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['val_auroc'].append(v_auroc)

        improvement = (v_auroc - 0.72) * 100
        print(f"\n{'─'*70}")
        print(f"EPOCH {epoch+1} SUMMARY:")
        print(f"  Train Loss: {t_loss:.4f}  |  Train Acc: {t_acc:.2f}%")
        print(f"  Val Loss:   {v_loss:.4f}  |  Val Acc:   {v_acc:.2f}%")
        print(f"  Val AUROC:  {v_auroc:.4f}  |  vs MRI baseline: {improvement:+.1f}%")
        print(f"{'─'*70}")

        scheduler.step(v_loss)

        if v_auroc > best_auroc + config.min_delta:
            best_auroc    = v_auroc
            best_val_loss = v_loss
            best_epoch    = epoch + 1
            epochs_without_improvement = 0

            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             v_loss,
                'val_auroc':            v_auroc,
            }, config.checkpoint_dir / "best_fusion_model.pth")
            print(f"\n✓ NEW BEST MODEL SAVED (epoch {epoch+1}) | AUROC: {v_auroc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"\n  No improvement for {epochs_without_improvement}/{config.patience} epochs")

        if epochs_without_improvement >= config.patience:
            print(f"\n{'='*70}\nEARLY STOPPING at epoch {epoch+1} | Best epoch: {best_epoch}\n{'='*70}")
            break

    training_time = datetime.now() - start_time

    # ── SAVE RESULTS ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("SAVING RESULTS")
    print("─" * 70)

    plot_training_curves(history, config.results_dir / "fusion_training_curves.png")

    checkpoint = torch.load(config.checkpoint_dir / "best_fusion_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    _, _, final_auroc, final_labels, final_probs = validate(
        model, val_loader, criterion, config.device, epoch=best_epoch - 1)

    plot_confusion_matrix(final_labels, final_probs,
                          save_path=config.results_dir / "fusion_confusion_matrix.png")

    val_preds = (np.array(final_probs) >= 0.5).astype(int)
    report = classification_report(final_labels, val_preds,
                                   target_names=['Benign', 'Cancer'], digits=4)

    with open(config.results_dir / "fusion_results.json", 'w') as f:
        json.dump({
            'training_time':             str(training_time),
            'best_epoch':                best_epoch,
            'total_epochs':              epoch + 1,
            'best_val_loss':             float(best_val_loss),
            'best_val_auroc':            float(best_auroc),
            'final_val_auroc':           float(final_auroc),
            'mri_baseline_auroc':        0.72,
            'improvement_over_baseline': float((final_auroc - 0.72) * 100),
        }, f, indent=2)

    # ── FINAL SUMMARY ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTraining Time:  {training_time}")
    print(f"Best Epoch:     {best_epoch}")
    print(f"Best Val AUROC: {best_auroc:.4f}")
    print(f"vs MRI baseline: {(best_auroc - 0.72)*100:+.1f}%")
    print(f"\nClassification Report:\n{report}")

    if best_auroc >= 0.77:
        print("\n✓ SUCCESS: Fusion target (0.77) achieved!")
    else:
        print(f"\n⚠ Below target: {best_auroc:.4f} < 0.77")

    print("\n" + "=" * 70)
    print("⚠ DOWNLOAD CHECKPOINT BEFORE STOPPING POD!")
    print(f"   {config.checkpoint_dir / 'best_fusion_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
