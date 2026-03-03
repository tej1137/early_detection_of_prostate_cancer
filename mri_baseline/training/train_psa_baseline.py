"""
training/train_psa_baseline.py
FIXED: Clinical-only dataset (no MRI loading), absolute paths,
       AUROC checkpointing, verbose removed, WeightedRandomSampler
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json
from datetime import datetime

from mri_baseline.models.psa_encoder import PSAEncoder, PSAClassifier
from mri_baseline.data.load_psa import (
    load_clinical_data,
    fit_normalisation,
    split_case_ids,
    CLINICAL_FEATURES,
)

# ══════════════════════════════════════════════════════════
# CLINICAL-ONLY DATASET  (no MRI, no .mha files)
# ══════════════════════════════════════════════════════════

class ClinicalDataset(Dataset):
    """
    Reads ONLY from the clinical DataFrame.
    No images are loaded — each sample is a 4-element feature vector.
    """
    def __init__(self, case_ids, clinical_df, norm_stats):
        self.case_ids    = case_ids
        self.clinical_df = clinical_df
        self.norm_stats  = norm_stats

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        row     = self.clinical_df.loc[case_id]

        # Z-score normalise
        features = np.array([
            (float(row[col]) - self.norm_stats[col]['mean']) / self.norm_stats[col]['std']
            for col in CLINICAL_FEATURES
        ], dtype=np.float32)

        label = int(row['case_csPCa'])
        return {
            'clinical': torch.tensor(features),
            'label':    torch.tensor(label, dtype=torch.long),
            'case_id':  case_id,
        }


def build_clinical_dataloaders(batch_size=16, num_workers=4):
    """Build train/val/test loaders with NO MRI loading."""
    project_root   = Path("/workspace/early_detection_of_prostate_cancer")
    marksheet_path = Path("/workspace/data/picai_labels/clinical_information/marksheet.csv")

    df = load_clinical_data(marksheet_path)
    train_ids, val_ids, test_ids = split_case_ids(df)

    # Norm stats from train set only (no leakage)
    norm_stats = fit_normalisation(df.loc[train_ids])

    train_ds = ClinicalDataset(train_ids, df, norm_stats)
    val_ds   = ClinicalDataset(val_ids,   df, norm_stats)
    test_ds  = ClinicalDataset(test_ids,  df, norm_stats)

    # Balanced sampler
    labels       = np.array([int(df.loc[cid, 'case_csPCa']) for cid in train_ids])
    class_counts  = np.bincount(labels)
    print(f"  Train class distribution: Benign={class_counts[0]}, Cancer={class_counts[1]}")
    class_weights  = 1.0 / class_counts
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.double)
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    return train_loader, val_loader, test_loader, norm_stats


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

class Config:
    # Model
    in_features   = 4
    embedding_dim = 256
    num_classes   = 2
    dropout       = 0.2

    # Training
    batch_size    = 16
    num_epochs    = 100
    learning_rate = 1e-3
    weight_decay  = 1e-4

    # Early stopping
    patience  = 20
    min_delta = 1e-4

    # Data
    num_workers = 4

    # FIXED: Absolute paths
    project_root   = Path("/workspace/early_detection_of_prostate_cancer")
    checkpoint_dir = project_root / "checkpoints"
    results_dir    = project_root / "results"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in pbar:
        clinical = batch['clinical'].to(device)
        labels   = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(clinical)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clinical.size(0)
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
            clinical = batch['clinical'].to(device)
            labels   = batch['label'].to(device)

            logits = model(clinical)
            loss   = criterion(logits, labels)
            probs  = torch.softmax(logits, dim=1)[:, 1]

            running_loss += loss.item() * clinical.size(0)
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

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'],   'r-', label='Val',   linewidth=2)
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'],   'r-', label='Val',   linewidth=2)
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['val_auroc'], 'g-', linewidth=2)
    axes[2].axhline(y=0.5,  color='gray',   linestyle='--', alpha=0.5, label='Random')
    axes[2].axhline(y=0.72, color='orange', linestyle='--', alpha=0.7, label='MRI baseline')
    axes[2].set_title('Val AUROC'); axes[2].set_ylim([0.4, 1.0])
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

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
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=20, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=14); ax.set_ylabel('True', fontsize=14)
    ax.set_title('Confusion Matrix - PSA Only', fontsize=16)
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
    print("PSA-ONLY BASELINE TRAINING (Fixed — Clinical Only)")
    print("=" * 70)

    config = Config()
    print(f"\nDevice: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. LOAD DATA (clinical only, no MRI) ──────────────
    print("\n" + "─" * 70)
    print("LOADING CLINICAL DATA (no MRI files)")
    print("─" * 70)

    train_loader, val_loader, test_loader, norm_stats = build_clinical_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # ── 2. MODEL ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("CREATING MODEL")
    print("─" * 70)

    model = PSAClassifier(
        in_features=config.in_features,
        embedding_dim=config.embedding_dim,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ PSA classifier | Parameters: {total_params:,}")

    # ── 3. TRAINING SETUP ─────────────────────────────────
    print("\n" + "─" * 70)
    print("TRAINING SETUP")
    print("─" * 70)

    class_weights = torch.tensor([1.0, 2.5]).to(config.device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)  # FIXED: no verbose=True

    print(f"✓ Loss: CrossEntropyLoss [benign=1.0, cancer=5.0]")
    print(f"✓ Optimizer: Adam (lr={config.learning_rate})")
    print(f"✓ Sampler: WeightedRandomSampler (balanced batches)")
    print(f"✓ Checkpointing: AUROC-based")
    print(f"✓ No MRI loading — epochs will be FAST (~seconds each)")

    # ── 4. TRAINING LOOP ──────────────────────────────────
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

        print(f"\n{'─'*70}")
        print(f"EPOCH {epoch+1} SUMMARY:")
        print(f"  Train Loss: {t_loss:.4f}  |  Train Acc: {t_acc:.2f}%")
        print(f"  Val Loss:   {v_loss:.4f}  |  Val Acc:   {v_acc:.2f}%")
        print(f"  Val AUROC:  {v_auroc:.4f}")
        print(f"{'─'*70}")

        scheduler.step(v_loss)

        # FIXED: Save on best AUROC
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
                'norm_stats':           norm_stats,
            }, config.checkpoint_dir / "best_psa_model.pth")
            print(f"\n✓ NEW BEST MODEL SAVED (epoch {epoch+1}) | AUROC: {v_auroc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"\n  No improvement for {epochs_without_improvement}/{config.patience} epochs")

        if epochs_without_improvement >= config.patience:
            print(f"\n{'='*70}\nEARLY STOPPING at epoch {epoch+1} | Best epoch: {best_epoch}\n{'='*70}")
            break

    training_time = datetime.now() - start_time

    # ── 5. SAVE RESULTS ───────────────────────────────────
    print("\n" + "─" * 70)
    print("SAVING RESULTS")
    print("─" * 70)

    plot_training_curves(history, config.results_dir / "psa_training_curves.png")

    checkpoint = torch.load(config.checkpoint_dir / "best_psa_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    _, _, final_auroc, final_labels, final_probs = validate(
        model, val_loader, criterion, config.device, epoch=best_epoch - 1)

    plot_confusion_matrix(final_labels, final_probs,
                          save_path=config.results_dir / "psa_confusion_matrix.png")

    val_preds = (np.array(final_probs) >= 0.5).astype(int)
    report = classification_report(final_labels, val_preds,
                                   target_names=['Benign', 'Cancer'], digits=4)

    with open(config.results_dir / "psa_results.json", 'w') as f:
        json.dump({
            'training_time':  str(training_time),
            'best_epoch':     best_epoch,
            'total_epochs':   epoch + 1,
            'best_val_loss':  float(best_val_loss),
            'best_val_auroc': float(best_auroc),
            'final_val_auroc': float(final_auroc),
        }, f, indent=2)

    # ── 6. FINAL SUMMARY ──────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTraining Time:  {training_time}")
    print(f"Best Epoch:     {best_epoch}")
    print(f"Best Val AUROC: {best_auroc:.4f}")
    print(f"\nClassification Report:\n{report}")

    if best_auroc >= 0.68:
        print("\n✓ SUCCESS: PSA baseline target (0.68) achieved!")
    else:
        print(f"\n⚠ Below target: {best_auroc:.4f} < 0.68")

    print("\nFiles saved:")
    print(f"  Model:     {config.checkpoint_dir / 'best_psa_model.pth'}")
    print(f"  Curves:    {config.results_dir / 'psa_training_curves.png'}")
    print(f"  Confusion: {config.results_dir / 'psa_confusion_matrix.png'}")
    print(f"  Results:   {config.results_dir / 'psa_results.json'}")
    print("=" * 70)
    print("⚠ DOWNLOAD CHECKPOINT BEFORE STOPPING POD!")
    print(f"   {config.checkpoint_dir / 'best_psa_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
