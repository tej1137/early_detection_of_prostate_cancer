"""
mri_baseline/training/train_fusion_clip.py

Fine-tuning CLIP-Pretrained Encoders for csPCa Classification.

Works for both Option A and Option B pretrained weights.
Points to the correct checkpoint directory based on --option_a / --option_b.


Run:
    python -m mri_baseline.training.train_fusion_clip --option_a --unfrozen
    python -m mri_baseline.training.train_fusion_clip --option_a --frozen
    python -m mri_baseline.training.train_fusion_clip --option_b --unfrozen
    python -m mri_baseline.training.train_fusion_clip --option_b --frozen
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

from mri_baseline.models.mri_encoder      import MRIEncoder
from mri_baseline.models.psa_encoder      import PSAEncoder
from mri_baseline.data.multimodal_dataset import PiCAIDataset, DataConfig


# Checkpoint dirs per option — set at runtime based on args
CKPT_DIRS = {
    "A": Path("/workspace/checkpoints/clip_a"),
    "B": Path("/workspace/checkpoints/clip_b"),
}

CFG = {
    "epochs"              : 50,
    "batch_size"          : 8,
    "lr_head"             : 1e-3,
    "lr_encoder"          : 1e-4,
    "weight_decay"        : 1e-4,
    "num_workers"         : 4,

    # IMPORTANT: clinical encoder is 256-dim for CLIP (vs 128 for crossmodal)
    "mri_encoder_dim"     : 512,
    "clinical_encoder_dim": 256,

    # Fusion input = 512 + 256 = 768
    "fusion_input_dim"    : 768,

    "results_dir"         : Path("/workspace/data/results/clip"),
}

class CLIPFusionModel(nn.Module):
    """
    CLIP-pretrained MRI + Clinical encoders + classification head.

    Works for both Option A and Option B — only difference is
    which checkpoint directory the weights come from.

    MRI    (B, 3, 20, 160, 160) → MRIEncoder(512)  → (B, 512)
    Clinical          (B, 4)   → PSAEncoder(256)  → (B, 256)
    Concatenate                                   → (B, 768)
    Fusion head                                   → (B, 256)
    Classifier                                    → (B, 2)

    Note: clinical_encoder_dim=256 here (wider than CrossModalFusionModel's 128)
    This is a DELIBERATE architectural difference between the two experiments.
    """

    def __init__(self, option: str, freeze_encoders: bool = False):
        super().__init__()

        ckpt_dir = CKPT_DIRS[option]

        # MRI encoder 
        self.mri_encoder = MRIEncoder(
            embedding_dim=CFG["mri_encoder_dim"]
        )
        mri_ckpt = ckpt_dir / "best_mri_encoder.pt"
        self.mri_encoder.load_state_dict(
            torch.load(mri_ckpt, map_location="cpu", weights_only=True)
        )
        print(f"   CLIP-{option} MRI encoder loaded from {mri_ckpt}")

        # Clinical encoder (256-dim — wider than crossmodal)
        self.clinical_encoder = PSAEncoder(
            in_features   = 4,
            embedding_dim = CFG["clinical_encoder_dim"],   # 256
        )
        clin_ckpt = ckpt_dir / "best_clinical_encoder.pt"
        self.clinical_encoder.load_state_dict(
            torch.load(clin_ckpt, map_location="cpu", weights_only=True)
        )
        print(f"   CLIP-{option} clinical encoder loaded from {clin_ckpt}")

        # Freeze encoders if linear probe mode─
        if freeze_encoders:
            for p in self.mri_encoder.parameters():
                p.requires_grad = False
            for p in self.clinical_encoder.parameters():
                p.requires_grad = False
            print(f"   Both encoders FROZEN — linear probe mode")
        else:
            print(f"   Both encoders UNFROZEN — full fine-tuning")

        # Fusion head 
        # Input is 768 (= 512 + 256) — wider than CrossModalFusionModel (640)
        self.fusion_head = nn.Sequential(
            nn.Linear(CFG["fusion_input_dim"], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(128, 2)
        self._init_new_layers()

    def _init_new_layers(self):
        for m in [self.fusion_head, self.classifier]:
            layers = m if isinstance(m, nn.Sequential) else [m]
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, mri: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        mri_feat  = self.mri_encoder(mri)                         # (B, 512)
        clin_feat = self.clinical_encoder(clinical)                # (B, 256)
        fused     = torch.cat([mri_feat, clin_feat], dim=1)       # (B, 768)
        fused     = self.fusion_head(fused)                        # (B, 128)
        return self.classifier(fused)                              # (B, 2)

    def predict_proba(self, mri: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(mri, clinical), dim=1)[:, 1]


# FOCAL LOSS

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# DATALOADERS

def build_dataloaders():
    data_cfg     = DataConfig()
    train_ds     = PiCAIDataset("train", data_cfg, augment=True)
    val_ds       = PiCAIDataset("val",   data_cfg, augment=False)
    test_ds      = PiCAIDataset("test",  data_cfg, augment=False)

    labels       = [int(train_ds.df.loc[cid, data_cfg.target_col])
                    for cid in train_ds.case_ids]
    class_counts = np.bincount(labels)
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

    print(f"  Train — Benign: {class_counts[0]}  Cancer: {class_counts[1]}")

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], sampler=sampler,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# EVALUATE

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


# PLOTS

def save_plots(history: dict, labels, preds, run_dir: Path, tag: str):
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True)
    ax2.plot(history["train_auroc"], label="Train")
    ax2.plot(history["val_auroc"],   label="Val")
    ax2.set_title("AUROC"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True)
    plt.suptitle(f"CLIP Fusion [{tag}]")
    plt.tight_layout()
    plt.savefig(run_dir / f"training_curves_{tag}.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Cancer"],
        yticklabels=["Benign", "Cancer"],
    )
    plt.title(f"CLIP Fusion [{tag}] — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(run_dir / f"confusion_matrix_{tag}.png", dpi=150)
    plt.close()


# TRAIN

def train(option: str, freeze_encoders: bool):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode    = "frozen" if freeze_encoders else "unfrozen"
    tag     = f"option{option}_{mode}"
    out_dir = CFG["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CLIP Fusion Fine-tuning  [Option {option} | {mode.upper()}]")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs     : {CFG['epochs']}")
    print(f"  Batch size : {CFG['batch_size']}")
    print(f"  Fusion dim : {CFG['fusion_input_dim']} (512 MRI + 256 clinical)")
    print(f"{'='*60}\n")

    model   = CLIPFusionModel(option=option, freeze_encoders=freeze_encoders).to(device)
    loss_fn = FocalLoss(gamma=2.0)

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

    ckpt_path = CKPT_DIRS[option] / f"fusion_clip_{tag}.pt"
    log_path  = CKPT_DIRS[option] / f"fusion_clip_{tag}_log.csv"

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_auroc",
            "val_loss", "val_acc", "val_auroc",
        ])

    history    = {"train_loss": [], "val_loss": [],
                  "train_auroc": [], "val_auroc": []}
    best_auroc = 0.0
    no_improve = 0

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        if freeze_encoders:
            model.mri_encoder.eval()
            model.clinical_encoder.eval()

        train_loss   = 0.0
        all_probs_t  = []
        all_labels_t = []
        t0           = time.time()

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

            train_loss   += loss.item()
            all_probs_t  += F.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            all_labels_t += labels.cpu().tolist()

        scheduler.step()

        avg_train   = train_loss / len(train_loader)
        train_auroc = roc_auc_score(all_labels_t, all_probs_t) \
                      if len(set(all_labels_t)) > 1 else 0.0
        val_loss, val_acc, val_auroc, _, _, _ = evaluate(
            model, val_loader, device, loss_fn
        )
        epoch_time = time.time() - t0

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

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"   Best model saved  (AUROC: {best_auroc:.4f})")
        else:
            no_improve += 1
            if no_improve >= 20:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # Test─
    print(f"\n{'='*60}")
    print(f"  Test Evaluation  [Option {option} | {mode.upper()}]")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    _, test_acc, test_auroc, preds, labels, probs = evaluate(
        model, test_loader, device, loss_fn
    )
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test AUROC    : {test_auroc:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['Benign','Cancer'])}")

    save_plots(history, labels, preds, out_dir, tag)

    results = {
        "model"          : f"CLIP-{option} Fusion [{mode}]",
        "option"         : option,
        "freeze_encoders": freeze_encoders,
        "best_val_auroc" : round(best_auroc,  4),
        "test_auroc"     : round(test_auroc,  4),
        "test_acc"       : round(test_acc,    4),
        "epochs_trained" : len(history["train_loss"]),
        "fusion_dim"     : CFG["fusion_input_dim"],
        "clinical_dim"   : CFG["clinical_encoder_dim"],
    }
    with open(out_dir / f"results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n   Results saved → {out_dir}")
    print(f"  Best Val AUROC : {best_auroc:.4f}")
    print(f"  Test AUROC     : {test_auroc:.4f}")
    print(f"{'='*60}\n")


# ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option_a",  action="store_true", help="Use CLIP-A weights")
    parser.add_argument("--option_b",  action="store_true", help="Use CLIP-B weights")
    parser.add_argument("--frozen",    action="store_true", help="Freeze encoders")
    parser.add_argument("--unfrozen",  action="store_true", help="Full fine-tuning")
    args = parser.parse_args()

    if args.option_a and args.option_b:
        print("ERROR: specify only one of --option_a or --option_b")
        exit(1)
    if not args.option_a and not args.option_b:
        print("No option specified — defaulting to --option_a")
        option = "A"
    else:
        option = "A" if args.option_a else "B"

    freeze = args.frozen and not args.unfrozen
    train(option=option, freeze_encoders=freeze)