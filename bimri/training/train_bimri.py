"""
bimri/training/train_bimri.py

Training script for bi-parametric MRI (T2W + ADC only) baseline.

Research question:
    Can biMRI (T2W + ADC) match mpMRI (T2W + ADC + HBV) performance?
    If yes → simpler acquisition protocol may suffice clinically.

Comparison target:
    mpMRI baseline (train_mri_baseline.py) → Test AUROC: 0.71
    biMRI baseline (this script)           → TBD

Architecture:
    BiMRIClassifier:
        BiMRIEncoder (in_channels=2) → (B, 512)
        Classification head          → (B, 2)

Everything else is IDENTICAL to train_mri_baseline.py:
    - Same focal loss (gamma=2.0)
    - Same weighted random sampler
    - Same learning rate (3e-5)
    - Same weight decay (1e-2)
    - Same early stopping (patience=20)
    - Same AUROC checkpointing

This ensures any AUROC difference is purely due to
losing the HBV channel, not any training differences.

Output:
    /workspace/data/results/bimri/run_001/
        best_bimri_model.pt
        bimri_training_curves.png
        bimri_confusion_matrix.png
        bimri_results.json
        params.json

Run:
    python -m bimri.training.train_bimri
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
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bimri.data.bimri_dataset     import get_bimri_loaders
from bimri.models.bimri_encoder   import BiMRIClassifier


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

class TrainConfig:
    """
    Identical to TrainConfig in train_mri_baseline.py.
    Only output_dir changes — everything else intentionally the same.
    """
    output_dir          = Path("/workspace/data/results/bimri")
    epochs              = 100
    batch_size          = 8
    learning_rate       = 3e-5
    weight_decay        = 1e-2
    lr_patience         = 7
    lr_factor           = 0.5
    early_stop_patience = 20
    focal_loss_gamma    = 2.0
    augment_train       = True
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed                = 42

    def __init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# EXPERIMENT TRACKING
# ══════════════════════════════════════════════════════════

def get_run_dir(base_dir: Path) -> Path:
    existing = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])
    run_dir = base_dir / f"run_{len(existing)+1:03d}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_params(run_dir: Path, cfg: TrainConfig):
    params = {
        "model"              : "biMRI Baseline (T2W + ADC only)",
        "channels"           : ["T2W", "ADC"],
        "dropped_channel"    : "HBV",
        "run_dir"            : str(run_dir),
        "epochs"             : cfg.epochs,
        "batch_size"         : cfg.batch_size,
        "learning_rate"      : cfg.learning_rate,
        "weight_decay"       : cfg.weight_decay,
        "lr_patience"        : cfg.lr_patience,
        "lr_factor"          : cfg.lr_factor,
        "early_stop_patience": cfg.early_stop_patience,
        "focal_loss_gamma"   : cfg.focal_loss_gamma,
        "augment_train"      : cfg.augment_train,
        "seed"               : cfg.seed,
        "mri_input_shape"    : [2, 20, 160, 160],   # ← 2 channels
    }
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"  ✓ Params saved → {run_dir / 'params.json'}")


# ══════════════════════════════════════════════════════════
# FOCAL LOSS  (identical to all other training scripts)
# ══════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ══════════════════════════════════════════════════════════
# TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []

    for batch in tqdm(loader, desc="  Training", leave=False):
        mri    = batch["mri"].to(device)
        labels = batch["label"].to(device)

        optimiser.zero_grad()
        logits = model(mri)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        all_probs.extend(
            torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        )
        all_labels.extend(labels.cpu().numpy())

    auroc = roc_auc_score(all_labels, all_probs) \
            if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auroc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            mri    = batch["mri"].to(device)
            labels = batch["label"].to(device)
            logits = model(mri)
            total_loss += criterion(logits, labels).item()
            all_probs.extend(
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            )
            all_labels.extend(labels.cpu().numpy())

    auroc = roc_auc_score(all_labels, all_probs) \
            if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auroc, all_labels, all_probs


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════

def save_training_curves(history: dict, run_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss — biMRI (T2W+ADC)")
    ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot(history["train_auroc"], label="Train")
    ax2.plot(history["val_auroc"],   label="Val")
    ax2.set_title("AUROC — biMRI (T2W+ADC)")
    ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "bimri_training_curves.png", dpi=150)
    plt.close()


def save_confusion_matrix(labels, preds, run_dir: Path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Cancer"],
        yticklabels=["Benign", "Cancer"],
    )
    plt.title("biMRI Baseline (T2W+ADC) — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(run_dir / "bimri_confusion_matrix.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = get_run_dir(cfg.output_dir)

    print("=" * 60)
    print("biMRI BASELINE TRAINING  (T2W + ADC only, HBV dropped)")
    print(f"  Run    : {run_dir.name}")
    print(f"  Device : {cfg.device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs : {cfg.epochs}")
    print("=" * 60)

    save_params(run_dir, cfg)

    # ── Data ───────────────────────────────────────────────
    loaders    = get_bimri_loaders(
        batch_size  = cfg.batch_size,
        num_workers = 4,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]

    # ── Model ──────────────────────────────────────────────
    model     = BiMRIClassifier(in_channels=2, dropout=0.5).to(cfg.device)
    criterion = FocalLoss(gamma=cfg.focal_loss_gamma)
    optimiser = optim.AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max",
        patience=cfg.lr_patience,
        factor=cfg.lr_factor,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_params:,}")
    print(f"  Input shape : (B, 2, 20, 160, 160)  ← 2 channels\n")

    # ── Training loop ──────────────────────────────────────
    history    = {"train_loss": [], "val_loss": [],
                  "train_auroc": [], "val_auroc": []}
    best_auroc = 0.0
    no_improve = 0
    best_ckpt  = run_dir / "best_bimri_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_auroc   = train_epoch(
            model, train_loader, optimiser, criterion, cfg.device
        )
        val_loss, val_auroc, _, _ = evaluate(
            model, val_loader, criterion, cfg.device
        )
        scheduler.step(val_auroc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auroc"].append(train_auroc)
        history["val_auroc"].append(val_auroc)

        print(
            f"  Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f}  AUROC: {train_auroc:.4f} | "
            f"Val Loss: {val_loss:.4f}  AUROC: {val_auroc:.4f}"
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_auroc"  : val_auroc,
                "params"     : json.load(open(run_dir / "params.json")),
            }, best_ckpt)
            print(f"    ✓ Best model saved (val AUROC: {val_auroc:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST EVALUATION  (biMRI — T2W + ADC only)")
    print("=" * 60)

    model.load_state_dict(
        torch.load(best_ckpt, weights_only=True)["model_state"]
    )
    _, test_auroc, test_labels, test_probs = evaluate(
        model, test_loader, criterion, cfg.device
    )
    test_preds = (np.array(test_probs) >= 0.5).astype(int)

    print(f"\n  Test AUROC : {test_auroc:.4f}")
    print(f"  (mpMRI baseline was 0.71 — difference shows HBV contribution)")
    print(classification_report(
        test_labels, test_preds, target_names=["Benign", "Cancer"]
    ))

    # ── Save outputs ───────────────────────────────────────
    save_training_curves(history, run_dir)
    save_confusion_matrix(test_labels, test_preds, run_dir)

    results = {
        "model"          : "biMRI Baseline (T2W+ADC only)",
        "channels"       : ["T2W", "ADC"],
        "dropped"        : "HBV",
        "run_dir"        : str(run_dir),
        "best_val_auroc" : round(best_auroc,  4),
        "test_auroc"     : round(test_auroc,  4),
        "epochs_trained" : len(history["train_loss"]),
        "timestamp"      : datetime.now().isoformat(),
        "comparison"     : {
            "mpMRI_baseline_auroc": 0.71,
            "difference"          : round(test_auroc - 0.71, 4),
        },
    }
    with open(run_dir / "bimri_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ All outputs saved → {run_dir}")
    print(f"  Best Val AUROC : {best_auroc:.4f}")
    print(f"  Test AUROC     : {test_auroc:.4f}")
    print(f"  mpMRI baseline : 0.7100")
    print(f"  HBV contribution: {test_auroc - 0.71:+.4f} AUROC points")
    print("=" * 60)


if __name__ == "__main__":
    main()