"""
mri_baseline/training/train_fusion_crossmodal.py

Fine-tuning Cross-Modal Pretrained Encoders for csPCa Classification.

What this script does:
    1. Loads BOTH pretrained encoders from cross-modal pretraining
       (best_mri_encoder.pt + best_clinical_encoder.pt)
    2. Attaches a fusion + classification head
    3. Fine-tunes on labelled train split
    4. Evaluates on val split per epoch
    5. Reports final test metrics + confusion matrix

Architecture:
    CrossModal-pretrained MRI Encoder → (B, 512)
    CrossModal-pretrained PSA Encoder → (B, 128)
    Concatenate                       → (B, 640)
    Fusion Head (Linear+ReLU+Dropout) → (B, 256)
    Classifier  (Linear+ReLU+Linear)  → (B, 2)

Two modes (same as train_fusion_contrastive.py for fair comparison):
    --frozen   : both encoders frozen → only head trains (linear probe)
    --unfrozen : full model trains end-to-end

Output:
    /workspace/checkpoints/crossmodal/
        fusion_crossmodal_frozen.pt
        fusion_crossmodal_unfrozen.pt
        fusion_crossmodal_frozen_log.csv
        fusion_crossmodal_unfrozen_log.csv

Run:
    python -m mri_baseline.training.train_fusion_crossmodal --frozen
    python -m mri_baseline.training.train_fusion_crossmodal --unfrozen
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import time
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from mri_baseline.models.mri_encoder      import MRIEncoder
from mri_baseline.models.psa_encoder      import PSAEncoder
from mri_baseline.data.multimodal_dataset import PiCAIDataset, DataConfig


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

CFG = {
    "epochs"           : 50,
    "batch_size"       : 8,
    "lr_head"          : 1e-3,     # LR for fusion head (always trains)
    "lr_encoder"       : 1e-4,     # LR for encoders (unfrozen mode only)
    "weight_decay"     : 1e-4,
    "num_workers"      : 4,

    # Encoder output dims — must match what was used during pretraining
    "mri_encoder_dim"  : 512,
    "clinical_encoder_dim": 128,

    # Fusion head input = mri_dim + clinical_dim
    "fusion_input_dim" : 640,      # 512 + 128

    # Pretrained weights
    "mri_ckpt"         : Path("/workspace/checkpoints/crossmodal/best_mri_encoder.pt"),
    "clinical_ckpt"    : Path("/workspace/checkpoints/crossmodal/best_clinical_encoder.pt"),

    # Output
    "ckpt_dir"         : Path("/workspace/checkpoints/crossmodal"),
    "results_dir"      : Path("/workspace/data/results/crossmodal"),
}


# ══════════════════════════════════════════════════════════
# FUSION MODEL
# ══════════════════════════════════════════════════════════

class CrossModalFusionModel(nn.Module):
    """
    Cross-modal pretrained MRI + Clinical encoders + classification head.

    MRI    (B, 3, 20, 160, 160) → MRIEncoder    → (B, 512)
    Clinical          (B, 4)   → PSAEncoder    → (B, 128)
    Concatenate                               → (B, 640)
    Fusion head                               → (B, 256)
    Classifier                                → (B, 2)

    Args:
        freeze_encoders: if True, both encoders are frozen (linear probe)
    """

    def __init__(self, freeze_encoders: bool = False):
        super().__init__()

        # ── MRI encoder — load cross-modal pretrained weights ──
        self.mri_encoder = MRIEncoder(embedding_dim=CFG["mri_encoder_dim"])
        mri_state = torch.load(
            CFG["mri_ckpt"], map_location="cpu", weights_only=True
        )
        self.mri_encoder.load_state_dict(mri_state)
        print(f"  ✓ Cross-modal MRI encoder loaded from {CFG['mri_ckpt']}")

        # ── Clinical encoder — load cross-modal pretrained weights ──
        self.clinical_encoder = PSAEncoder(
            in_features   = 4,
            embedding_dim = CFG["clinical_encoder_dim"],
        )
        clin_state = torch.load(
            CFG["clinical_ckpt"], map_location="cpu", weights_only=True
        )
        self.clinical_encoder.load_state_dict(clin_state)
        print(f"  ✓ Cross-modal clinical encoder loaded from {CFG['clinical_ckpt']}")

        # ── Optionally freeze both encoders ───────────────
        if freeze_encoders:
            for param in self.mri_encoder.parameters():
                param.requires_grad = False
            for param in self.clinical_encoder.parameters():
                param.requires_grad = False
            print(f"  ✓ Both encoders FROZEN — training head only")
        else:
            print(f"  ✓ Both encoders UNFROZEN — full fine-tuning")

        # ── Fusion head ────────────────────────────────────
        self.fusion_head = nn.Sequential(
            nn.Linear(CFG["fusion_input_dim"], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # ── Classifier ─────────────────────────────────────
        self.classifier = nn.Linear(128, 2)

        # ── Initialise new layers ──────────────────────────
        self._init_new_layers()

    def _init_new_layers(self):
        for m in [self.fusion_head, self.classifier]:
            for layer in (m if isinstance(m, nn.Sequential) else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, mri: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mri     : (B, 3, 20, 160, 160)
            clinical: (B, 4)
        Returns:
            logits  : (B, 2)
        """
        mri_feat  = self.mri_encoder(mri)              # (B, 512)
        clin_feat = self.clinical_encoder(clinical)     # (B, 128)
        fused     = torch.cat([mri_feat, clin_feat], dim=1)  # (B, 640)
        fused     = self.fusion_head(fused)             # (B, 128)
        return self.classifier(fused)                   # (B, 2)

    def predict_proba(self, mri: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        """Returns P(cancer) for AUROC calculation."""
        return F.softmax(self.forward(mri, clinical), dim=1)[:, 1]


# ══════════════════════════════════════════════════════════
# DATALOADERS
# ══════════════════════════════════════════════════════════

def build_dataloaders():
    data_cfg = DataConfig()

    train_ds = PiCAIDataset("train", data_cfg, augment=True)
    val_ds   = PiCAIDataset("val",   data_cfg, augment=False)
    test_ds  = PiCAIDataset("test",  data_cfg, augment=False)

    # Weighted sampler for class imbalance (~72% benign, ~28% cancer)
    labels       = [int(train_ds.df.loc[cid, data_cfg.target_col])
                    for cid in train_ds.case_ids]
    class_counts = np.bincount(labels)
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )

    print(f"  Train — Benign: {class_counts[0]}  Cancer: {class_counts[1]}")

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        sampler=sampler, num_workers=CFG["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=CFG["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=CFG["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════
# FOCAL LOSS  (same as baseline scripts for fair comparison)
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
# EVALUATE
# ══════════════════════════════════════════════════════════

def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.no_grad():
        for batch in loader:
            mri      = batch["mri"].to(device,      non_blocking=True)
            clinical = batch["clinical"].to(device,  non_blocking=True)
            labels   = batch["label"].to(device,     non_blocking=True)

            with autocast():
                logits = model(mri, clinical)
                loss   = loss_fn(logits, labels)

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            probs       = F.softmax(logits, dim=1)[:, 1]
            all_preds  += preds.cpu().tolist()
            all_labels += labels.cpu().tolist()
            all_probs  += probs.cpu().tolist()

    avg_loss = total_loss / len(loader)
    acc      = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    auroc    = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auroc, all_preds, all_labels, all_probs


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════

def save_confusion_matrix(labels, preds, run_dir: Path, mode: str):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Cancer"],
        yticklabels=["Benign", "Cancer"],
    )
    plt.title(f"CrossModal Fusion [{mode.upper()}] — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(run_dir / f"confusion_matrix_{mode}.png", dpi=150)
    plt.close()


def save_training_curves(history: dict, run_dir: Path, mode: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True)
    ax2.plot(history["train_auroc"], label="Train")
    ax2.plot(history["val_auroc"],   label="Val")
    ax2.set_title("AUROC"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True)
    plt.suptitle(f"CrossModal Fusion [{mode.upper()}]")
    plt.tight_layout()
    plt.savefig(run_dir / f"training_curves_{mode}.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════

def train(freeze_encoders: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode   = "frozen" if freeze_encoders else "unfrozen"

    print(f"\n{'='*60}")
    print(f"  CrossModal Fusion Fine-tuning  [{mode.upper()}]")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs     : {CFG['epochs']}")
    print(f"  Batch size : {CFG['batch_size']}")
    print(f"  Mode       : {'Encoders FROZEN (linear probe)' if freeze_encoders else 'Full fine-tuning'}")
    print(f"{'='*60}\n")

    CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    CFG["results_dir"].mkdir(parents=True, exist_ok=True)

    model   = CrossModalFusionModel(freeze_encoders=freeze_encoders).to(device)
    loss_fn = FocalLoss(gamma=2.0)

    # ── Optimiser — separate LRs for encoders vs head ─────
    if freeze_encoders:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CFG["lr_head"], weight_decay=CFG["weight_decay"],
        )
    else:
        optimizer = optim.AdamW([
            {"params": model.mri_encoder.parameters(),      "lr": CFG["lr_encoder"]},
            {"params": model.clinical_encoder.parameters(), "lr": CFG["lr_encoder"]},
            {"params": model.fusion_head.parameters(),      "lr": CFG["lr_head"]},
            {"params": model.classifier.parameters(),       "lr": CFG["lr_head"]},
        ], weight_decay=CFG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6
    )
    scaler = GradScaler()

    train_loader, val_loader, test_loader = build_dataloaders()

    # ── Log file ───────────────────────────────────────────
    log_path = CFG["ckpt_dir"] / f"fusion_crossmodal_{mode}_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_auroc",
            "val_loss", "val_acc", "val_auroc"
        ])

    history    = {"train_loss": [], "val_loss": [],
                  "train_auroc": [], "val_auroc": []}
    best_auroc = 0.0
    no_improve = 0
    ckpt_path  = CFG["ckpt_dir"] / f"fusion_crossmodal_{mode}.pt"

    for epoch in range(1, CFG["epochs"] + 1):
        # ── Train epoch ────────────────────────────────────
        model.train()
        if freeze_encoders:
            model.mri_encoder.eval()
            model.clinical_encoder.eval()

        train_loss  = 0.0
        all_probs_t = []
        all_labels_t = []
        t0 = time.time()

        for batch in train_loader:
            mri      = batch["mri"].to(device,      non_blocking=True)
            clinical = batch["clinical"].to(device,  non_blocking=True)
            labels   = batch["label"].to(device,     non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                logits = model(mri, clinical)
                loss   = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item()
            all_probs_t   += F.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            all_labels_t  += labels.cpu().tolist()

        scheduler.step()

        avg_train      = train_loss / len(train_loader)
        train_auroc    = roc_auc_score(all_labels_t, all_probs_t) \
                         if len(set(all_labels_t)) > 1 else 0.0
        val_loss, val_acc, val_auroc, _, _, _ = evaluate(
            model, val_loader, device, loss_fn
        )
        epoch_time = time.time() - t0

        # ── Logging ────────────────────────────────────────
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["train_auroc"].append(train_auroc)
        history["val_auroc"].append(val_auroc)

        print(
            f"  Epoch [{epoch:>3}/{CFG['epochs']}]  "
            f"Train: {avg_train:.4f} (AUROC {train_auroc:.3f})  "
            f"Val: {val_loss:.4f}  Acc: {val_acc:.3f}  AUROC: {val_auroc:.3f}  "
            f"({epoch_time:.1f}s)"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                round(avg_train,   6), round(train_auroc, 4),
                round(val_loss,    6), round(val_acc,     4),
                round(val_auroc,   4),
            ])

        # ── Checkpointing ──────────────────────────────────
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Best model saved  (AUROC: {best_auroc:.4f})")
        else:
            no_improve += 1
            if no_improve >= 20:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Test Evaluation  [{mode.upper()}]")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    _, test_acc, test_auroc, preds, labels, probs = evaluate(
        model, test_loader, device, loss_fn
    )
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test AUROC    : {test_auroc:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['Benign','Cancer'])}")

    # ── Save plots ─────────────────────────────────────────
    save_training_curves(history, CFG["results_dir"], mode)
    save_confusion_matrix(labels, preds, CFG["results_dir"], mode)

    # ── Save results JSON ──────────────────────────────────
    results = {
        "model"         : f"CrossModal Fusion [{mode}]",
        "best_val_auroc": round(best_auroc, 4),
        "test_auroc"    : round(test_auroc, 4),
        "test_acc"      : round(test_acc,   4),
        "epochs_trained": len(history["train_loss"]),
        "freeze_encoders": freeze_encoders,
    }
    with open(CFG["results_dir"] / f"results_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ Results saved → {CFG['results_dir']}")
    print(f"  Best Val AUROC : {best_auroc:.4f}")
    print(f"  Test AUROC     : {test_auroc:.4f}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen",   action="store_true",
        help="Freeze both encoders (linear probe)"
    )
    parser.add_argument(
        "--unfrozen", action="store_true",
        help="Full fine-tuning — all weights update"
    )
    args = parser.parse_args()

    if not args.frozen and not args.unfrozen:
        print("  No mode specified — defaulting to --unfrozen")
        freeze = False
    else:
        freeze = args.frozen and not args.unfrozen

    train(freeze_encoders=freeze)