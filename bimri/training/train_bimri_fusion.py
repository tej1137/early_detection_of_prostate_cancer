"""
Train biMRI + PSA clinical features fusion model.
Two modes:
    --scratch  : BiMRI encoder randomly initialised 
    --pretrained: Load BiMRI encoder from best_bimri_model.pt

Run:
    python -m bimri.training.train_bimri_fusion --scratch
    python -m bimri.training.train_bimri_fusion --pretrained
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
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bimri.models.bimri_encoder               import BiMRIEncoder
from mri_baseline.models.psa_encoder          import PSAEncoder
from mri_baseline.data.multimodal_dataset     import PiCAIDataset, DataConfig


CFG = {
    "epochs"              : 75,
    "batch_size"          : 8,
    "lr_head"             : 1e-3,
    "lr_encoder"          : 3e-5,     #same as train_mri_baseline.py
    "weight_decay"        : 1e-2,
    "num_workers"         : 4,
    "focal_loss_gamma"    : 2.0,
    "early_stop_patience" : 20,

    #Encoder dims
    "mri_encoder_dim"     : 512,
    "psa_encoder_dim"     : 256,
    "fusion_input_dim"    : 768,      #512 + 256

    #Pretrained biMRI encoder path
    "bimri_ckpt"          : Path("/workspace/data/results/bimri/run_001/best_bimri_model.pt"),

    #Output
    "ckpt_dir"            : Path("/workspace/checkpoints/bimri"),
    "results_dir"         : Path("/workspace/data/results/bimri"),
}


#Fusion model with BiMRI encoder + PSA encoder + fusion head + classifier

class BiMRIFusionModel(nn.Module):
    """
    biMRI (T2W+ADC) + PSA clinical features fusion model.

    BiMRIEncoder (2ch) → (B, 512)
    PSAEncoder         → (B, 256)
    Concat             → (B, 768)
    Fusion head        → (B, 256)
    Classifier         → (B, 2)

    Args:
        use_pretrained: load BiMRI encoder from best_bimri_model.pt
    """

    def __init__(self, use_pretrained: bool = False):
        super().__init__()

        #BiMRI encoder (2 channels)
        self.mri_encoder = BiMRIEncoder(
            in_channels   = 2,
            embedding_dim = CFG["mri_encoder_dim"],
            dropout       = 0.3,
        )

        if use_pretrained:
            if not CFG["bimri_ckpt"].exists():
                raise FileNotFoundError(
                    f"biMRI checkpoint not found: {CFG['bimri_ckpt']}\n"
                    f"Run train_bimri.py first."
                )
            ckpt = torch.load(
                CFG["bimri_ckpt"], map_location="cpu", weights_only=True
            )
            #Checkpoint saves full model state  extract encoder weights
            full_state  = ckpt["model_state"]
            encoder_state = {
                k.replace("encoder.", "", 1): v
                for k, v in full_state.items()
                if k.startswith("encoder.")
            }
            self.mri_encoder.load_state_dict(encoder_state)
            print(f"  ✓ Pretrained biMRI encoder loaded from {CFG['bimri_ckpt']}")
        else:
            print(f"  ✓ biMRI encoder randomly initialised (from scratch)")

        #PSA encoder
        self.psa_encoder = PSAEncoder(
            in_features   = 4,
            embedding_dim = CFG["psa_encoder_dim"],
            dropout       = 0.2,
        )
        print(f"  ✓ PSA encoder randomly initialised")

        #Fusion head
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
        """
        Args:
            mri     : (B, 2, 20, 160, 160)  ← 2 channels only
            clinical: (B, 4)
        Returns:
            logits  : (B, 2)
        """
        #Slice MRI
        if mri.shape[1] == 3:
            mri = mri[:, :2]   # drop HBV

        mri_feat  = self.mri_encoder(mri)                       # (B, 512)
        psa_feat  = self.psa_encoder(clinical)                  # (B, 256)
        fused     = torch.cat([mri_feat, psa_feat], dim=1)      # (B, 768)
        fused     = self.fusion_head(fused)                     # (B, 128)
        return self.classifier(fused)                           # (B, 2)

    def predict_proba(self, mri, clinical):
        return F.softmax(self.forward(mri, clinical), dim=1)[:, 1]


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

#Use PiCAIDataset returns full 3ch MRI, we slice to 2ch in model
def build_dataloaders():
    data_cfg = DataConfig()

    train_ds = PiCAIDataset("train", data_cfg, augment=True)
    val_ds   = PiCAIDataset("val",   data_cfg, augment=False)
    test_ds  = PiCAIDataset("test",  data_cfg, augment=False)

    labels       = [int(train_ds.df.loc[cid, data_cfg.target_col])
                    for cid in train_ds.case_ids]
    class_counts = np.bincount(labels)
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )

    print(f"  Train — Benign: {class_counts[0]}  Cancer: {class_counts[1]}")

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], sampler=sampler,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader, test_loader

#Evaluatee

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

#Plot cheyuu

def save_plots(history, labels, preds, results_dir, tag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"],  label="Train")
    ax1.plot(history["val_loss"],    label="Val")
    ax1.set_title(f"Loss — biMRI Fusion [{tag}]")
    ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot(history["train_auroc"], label="Train")
    ax2.plot(history["val_auroc"],   label="Val")
    ax2.set_title(f"AUROC — biMRI Fusion [{tag}]")
    ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"fusion_training_curves_{tag}.png", dpi=150)
    plt.close()

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Cancer"],
        yticklabels=["Benign", "Cancer"],
    )
    plt.title(f"biMRI Fusion [{tag}] — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(results_dir / f"fusion_confusion_matrix_{tag}.png", dpi=150)
    plt.close()

#trainnn cheyuu

def train(use_pretrained: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag    = "pretrained" if use_pretrained else "scratch"

    print(f"\n{'='*60}")
    print(f"  biMRI Fusion Training  [{tag.upper()}]")
    print(f"  T2W + ADC + PSA clinical features")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs     : {CFG['epochs']}")
    print(f"  Batch size : {CFG['batch_size']}")
    print(f"  MRI input  : 2 channels (T2W + ADC, HBV dropped)")
    print(f"  Fusion dim : {CFG['fusion_input_dim']} (512 MRI + 256 PSA)")
    print(f"{'='*60}\n")

    CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    CFG["results_dir"].mkdir(parents=True, exist_ok=True)

    model   = BiMRIFusionModel(use_pretrained=use_pretrained).to(device)
    loss_fn = FocalLoss(gamma=CFG["focal_loss_gamma"])

    optimizer = optim.AdamW([
        {"params": model.mri_encoder.parameters(), "lr": CFG["lr_encoder"]},
        {"params": model.psa_encoder.parameters(), "lr": CFG["lr_head"]},
        {"params": model.fusion_head.parameters(), "lr": CFG["lr_head"]},
        {"params": model.classifier.parameters(),  "lr": CFG["lr_head"]},
    ], weight_decay=CFG["weight_decay"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5
    )
    scaler = GradScaler()

    train_loader, val_loader, test_loader = build_dataloaders()

    ckpt_path = CFG["ckpt_dir"] / f"fusion_bimri_{tag}.pt"
    log_path  = CFG["ckpt_dir"] / f"fusion_bimri_{tag}_log.csv"

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

        avg_train   = train_loss / len(train_loader)
        train_auroc = roc_auc_score(all_labels_t, all_probs_t) \
                      if len(set(all_labels_t)) > 1 else 0.0
        val_loss, val_acc, val_auroc, _, _, _ = evaluate(
            model, val_loader, device, loss_fn
        )
        epoch_time = time.time() - t0

        scheduler.step(val_auroc)

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
            print(f"Best model saved  (AUROC: {best_auroc:.4f})")
        else:
            no_improve += 1
            if no_improve >= CFG["early_stop_patience"]:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    #Test 
    print(f"\n{'='*60}")
    print(f"  Test Evaluation  [biMRI Fusion {tag.upper()}]")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    _, test_acc, test_auroc, preds, labels, _ = evaluate(
        model, test_loader, device, loss_fn
    )

    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test AUROC    : {test_auroc:.4f}")
    print(f"\n  Comparison:")
    print(f"    biMRI alone          : 0.6745")
    print(f"    biMRI + PSA ({tag:>10}): {test_auroc:.4f}  ({test_auroc-0.6745:+.4f})")
    print(f"    mpMRI alone          : 0.7100")
    print(f"    mpMRI + PSA fusion   : 0.7823")
    print(f"\n{classification_report(labels, preds, target_names=['Benign','Cancer'])}")

    save_plots(history, labels, preds, CFG["results_dir"], tag)

    results = {
        "model"              : f"biMRI Fusion [{tag}]",
        "channels"           : ["T2W", "ADC"],
        "dropped"            : "HBV",
        "use_pretrained"     : use_pretrained,
        "best_val_auroc"     : round(best_auroc,  4),
        "test_auroc"         : round(test_auroc,  4),
        "test_acc"           : round(test_acc,    4),
        "epochs_trained"     : len(history["train_loss"]),
        "timestamp"          : datetime.now().isoformat(),
        "comparison"         : {
            "bimri_alone"        : 0.6745,
            "mpmri_alone"        : 0.7100,
            "mpmri_fusion"       : 0.7823,
            "bimri_fusion_gain"  : round(test_auroc - 0.6745, 4),
            "gap_vs_mpmri_fusion": round(test_auroc - 0.7823, 4),
        },
    }
    with open(CFG["results_dir"] / f"fusion_results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved → {CFG['results_dir']}")
    print(f"{'='*60}\n")

#Run the training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch", action="store_true",
        help="Train biMRI encoder from scratch (default)"
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Load pretrained biMRI encoder from best_bimri_model.pt"
    )
    args = parser.parse_args()

    use_pretrained = args.pretrained and not args.scratch
    train(use_pretrained=use_pretrained)