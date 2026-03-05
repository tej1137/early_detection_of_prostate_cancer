"""
training/train_psa_baseline.py

PSA-only baseline training using preprocessed clinical data.
Uses multimodal_dataset.py with clinical_only=True — no MRI loading.

Run:
  python -m mri_baseline.training.train_psa_baseline
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from torch.utils.data import WeightedRandomSampler, DataLoader
from datetime import datetime

from mri_baseline.models.psa_encoder import PSAClassifier
from mri_baseline.data.multimodal_dataset import PiCAIDataset, DataConfig


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

class TrainConfig:
    output_dir           = Path("/workspace/data/results/psa_baseline")
    epochs               = 200
    batch_size           = 32
    learning_rate        = 1e-3
    weight_decay         = 1e-4
    lr_patience          = 7
    lr_factor            = 0.5
    early_stop_patience  = 30
    use_weighted_sampler = True
    focal_loss_gamma     = 2.0
    device               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed                 = 42

    def __init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# EXPERIMENT TRACKING
# ══════════════════════════════════════════════════════════

def get_run_dir(base_dir: Path) -> Path:
    """Auto-increment run folder: run_001, run_002..."""
    existing = sorted([d for d in base_dir.iterdir()
                       if d.is_dir() and d.name.startswith("run_")])
    run_dir = base_dir / f"run_{len(existing)+1:03d}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_params(run_dir: Path, train_cfg: TrainConfig, data_cfg: DataConfig):
    params = {
        "model"               : "PSA Baseline",
        "run_dir"             : str(run_dir),
        "epochs"              : train_cfg.epochs,
        "batch_size"          : train_cfg.batch_size,
        "learning_rate"       : train_cfg.learning_rate,
        "weight_decay"        : train_cfg.weight_decay,
        "lr_patience"         : train_cfg.lr_patience,
        "lr_factor"           : train_cfg.lr_factor,
        "early_stop_patience" : train_cfg.early_stop_patience,
        "use_weighted_sampler": train_cfg.use_weighted_sampler,
        "focal_loss_gamma"    : train_cfg.focal_loss_gamma,
        "seed"                : train_cfg.seed,
        "clinical_features"   : data_cfg.clinical_features,
        "augment_train"       : data_cfg.augment_train,
    }
    with open(run_dir / "params.json", 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  ✓ Params saved → {run_dir / 'params.json'}")


# ══════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ══════════════════════════════════════════════════════════
# DATALOADERS
# ══════════════════════════════════════════════════════════

def build_dataloaders(train_cfg: TrainConfig, data_cfg: DataConfig):
    train_ds = PiCAIDataset("train", data_cfg, augment=False, clinical_only=True)
    val_ds   = PiCAIDataset("val",   data_cfg, augment=False, clinical_only=True)
    test_ds  = PiCAIDataset("test",  data_cfg, augment=False, clinical_only=True)

    labels       = [int(train_ds.df.loc[cid, data_cfg.target_col]) for cid in train_ds.case_ids]
    class_counts = np.bincount(labels)
    print(f"  Train — Benign: {class_counts[0]}  Cancer: {class_counts[1]}")

    if train_cfg.use_weighted_sampler:
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                                  sampler=sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=train_cfg.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════
# TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    for batch in loader:
        clinical = batch["clinical"].to(device)
        labels   = batch["label"].to(device)
        optimiser.zero_grad()
        logits = model(clinical)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        all_probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auroc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device)
            labels   = batch["label"].to(device)
            logits   = model(clinical)
            total_loss += criterion(logits, labels).item()
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auroc, all_labels, all_probs


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════

def save_training_curves(history, run_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot(history["train_auroc"], label="Train")
    ax2.plot(history["val_auroc"],   label="Val")
    ax2.set_title("AUROC"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "psa_training_curves.png", dpi=150)
    plt.close()


def save_confusion_matrix(labels, preds, run_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Benign", "Cancer"],
                yticklabels=["Benign", "Cancer"])
    plt.title("PSA Baseline — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(run_dir / "psa_confusion_matrix.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    train_cfg = TrainConfig()
    data_cfg  = DataConfig()
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    run_dir = get_run_dir(train_cfg.output_dir)

    print("=" * 60)
    print("PSA BASELINE TRAINING")
    print(f"  Run    : {run_dir.name}")
    print(f"  Device : {train_cfg.device}")
    print(f"  Epochs : {train_cfg.epochs}")
    print("=" * 60)

    save_params(run_dir, train_cfg, data_cfg)

    train_loader, val_loader, test_loader = build_dataloaders(train_cfg, data_cfg)

    model     = PSAClassifier(in_features=4).to(train_cfg.device)
    criterion = FocalLoss(gamma=train_cfg.focal_loss_gamma)
    optimiser = optim.AdamW(model.parameters(),
                            lr=train_cfg.learning_rate,
                            weight_decay=train_cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='max', patience=train_cfg.lr_patience,
        factor=train_cfg.lr_factor)

    print(f"\n  Model params: {sum(p.numel() for p in model.parameters()):,}")

    history    = {"train_loss": [], "val_loss": [], "train_auroc": [], "val_auroc": []}
    best_auroc = 0.0
    no_improve = 0
    best_ckpt  = run_dir / "best_psa_model.pt"

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_auroc        = train_epoch(model, train_loader, optimiser, criterion, train_cfg.device)
        val_loss, val_auroc, _, _      = evaluate(model, val_loader, criterion, train_cfg.device)
        scheduler.step(val_auroc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auroc"].append(train_auroc)
        history["val_auroc"].append(val_auroc)

        print(f"  Epoch {epoch:02d}/{train_cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} AUROC: {train_auroc:.4f} | "
              f"Val Loss: {val_loss:.4f} AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_auroc"  : val_auroc,
                "norm_stats" : json.load(open(data_cfg.norm_stats)),
                "params"     : json.load(open(run_dir / "params.json")),
            }, best_ckpt)
            print(f"    ✓ Best model saved (val AUROC: {val_auroc:.4f})")
        else:
            no_improve += 1
            if no_improve >= train_cfg.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Test ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    model.load_state_dict(torch.load(best_ckpt, weights_only=True)["model_state"])
    _, test_auroc, test_labels, test_probs = evaluate(model, test_loader, criterion, train_cfg.device)
    test_preds = (np.array(test_probs) >= 0.5).astype(int)

    print(f"\n  Test AUROC: {test_auroc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=["Benign", "Cancer"]))

    save_training_curves(history, run_dir)
    save_confusion_matrix(test_labels, test_preds, run_dir)

    results = {
        "model"         : "PSA Baseline",
        "run_dir"       : str(run_dir),
        "best_val_auroc": round(best_auroc, 4),
        "test_auroc"    : round(test_auroc, 4),
        "epochs_trained": len(history["train_loss"]),
        "timestamp"     : datetime.now().isoformat(),
    }
    with open(run_dir / "psa_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ All outputs saved → {run_dir}")
    print(f"  Best Val AUROC : {best_auroc:.4f}")
    print(f"  Test AUROC     : {test_auroc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()