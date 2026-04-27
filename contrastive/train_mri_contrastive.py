"""
contrastive/train_mri_contrastive.py

Fine-tuning the contrastive pretrained MRI encoder for cancer classification.

    1. Loads pretrained encoder weights from best_encoder.pt
    2. Attaches a classification head
    3. Fine-tunes on labelled train split
    4. Evaluates on val split per epoch
    5. Reports final test metrics

Two modes:
    --frozen   : encoder frozen, only classification head trains (linear probe)
    --unfrozen : full model trains end-to-end (full fine-tuning)

Output:
    /workspace/checkpoints/contrastive/
        mri_finetuned_frozen.pt    ← best frozen model
        mri_finetuned_unfrozen.pt  ← best unfrozen model
        mri_finetune_log.csv       ← metrics per epoch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import csv
import time
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mri_baseline.models.mri_encoder  import MRIEncoder
from mri_baseline.data.multimodal_dataset import PiCAIDataset

CFG = {
    "epochs"          : 50,
    "batch_size"      : 8,
    "lr_head"         : 1e-3,    # LR for classification head
    "lr_encoder"      : 1e-4,    # LR for encoder (unfrozen mode only)
    "weight_decay"    : 1e-4,
    "num_workers"     : 4,
    "encoder_dim"     : 512,

    "pretrained_path" : Path("/workspace/checkpoints/contrastive/best_encoder.pt"),
    "ckpt_dir"        : Path("/workspace/checkpoints/contrastive"),
}


#Classification head definition Simple MLP

class ClassificationHead(nn.Module):
    """
    Simple MLP head attached to pretrained encoder.
    512 → 128 → 2
    """
    def __init__(self, in_dim: int = 512, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

#full model definition
class FineTunedMRIModel(nn.Module):
    """Pretrained encoder + classification head."""

    def __init__(self, encoder: MRIEncoder, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head    = ClassificationHead(in_dim=CFG["encoder_dim"])

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)       # (B, 512)
        return self.head(features)       # (B, 2)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)[:, 1]


#Helper fuctionssss

def get_loaders():
    train_ds = PiCAIDataset(split="train", mri_only=True)
    val_ds   = PiCAIDataset(split="val",   mri_only=True)
    test_ds  = PiCAIDataset(split="test",  mri_only=True)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=CFG["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"],
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"],
                              pin_memory=True)
    return train_loader, val_loader, test_loader



def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []

    with torch.no_grad():
        for batch in loader:
            mri = batch["mri"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with autocast():
                logits = model(mri)
                loss   = loss_fn(logits, labels)
            total_loss  += loss.item()
            preds        = logits.argmax(dim=1)
            probs        = F.softmax(logits, dim=1)[:, 1]
            all_preds   += preds.cpu().tolist()
            all_labels  += labels.cpu().tolist()
            all_probs   += probs.cpu().tolist()

    avg_loss = total_loss / len(loader)
    acc      = np.mean(np.array(all_preds) == np.array(all_labels))
    auroc    = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auroc, all_preds, all_labels


#Train karooo

def train(freeze_encoder: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode   = "frozen" if freeze_encoder else "unfrozen"

    print(f"\n{'='*55}")
    print(f"  Contrastive MRI Fine-tuning  [{mode.upper()}]")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {CFG['epochs']}")
    print(f"  Batch size  : {CFG['batch_size']}")
    print(f"  Mode        : {'Encoder FROZEN (linear probe)' if freeze_encoder else 'Full fine-tuning'}")
    print(f"{'='*55}\n")

    CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    # Load pretrained encoder
    encoder = MRIEncoder(embedding_dim=CFG["encoder_dim"])
    state   = torch.load(CFG["pretrained_path"], map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    print(f"  Pretrained encoder loaded from {CFG['pretrained_path']}\n")

    model   = FineTunedMRIModel(encoder, freeze_encoder=freeze_encoder).to(device)

    # Class weights (same as baseline)
    class_weights = torch.tensor([1.0, 2.0], device=device)
    loss_fn       = nn.CrossEntropyLoss(weight=class_weights)

    # Optimiser — different LRs for encoder vs head
    if freeze_encoder:
        optimizer = optim.AdamW(
            model.head.parameters(),
            lr=CFG["lr_head"], weight_decay=CFG["weight_decay"]
        )
    else:
        optimizer = optim.AdamW([
            {"params": model.encoder.parameters(), "lr": CFG["lr_encoder"]},
            {"params": model.head.parameters(),    "lr": CFG["lr_head"]},
        ], weight_decay=CFG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6
    )
    scaler = GradScaler()

    train_loader, val_loader, test_loader = get_loaders()

    # Log file
    log_path = CFG["ckpt_dir"] / f"mri_finetune_{mode}_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "val_acc", "val_auroc"
        ])

    best_auroc = 0.0
    ckpt_path  = CFG["ckpt_dir"] / f"mri_finetuned_{mode}.pt"

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        if freeze_encoder:
            model.encoder.eval()   # keep BN frozen in linear probe mode

        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            mri    = batch["mri"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                logits = model(mri)
                loss   = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        avg_train = train_loss / len(train_loader)
        val_loss, val_acc, val_auroc, _, _ = evaluate(model, val_loader, device, loss_fn)
        epoch_time = time.time() - t0

        print(f"  Epoch [{epoch:>3}/{CFG['epochs']}]  "
              f"Train: {avg_train:.4f}  "
              f"Val: {val_loss:.4f}  "
              f"Acc: {val_acc:.3f}  "
              f"AUROC: {val_auroc:.3f}  "
              f"({epoch_time:.1f}s)")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, round(avg_train, 6), round(val_loss, 6),
                round(val_acc, 4), round(val_auroc, 4)
            ])

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Best model saved (AUROC: {best_auroc:.3f})")

    # Test evaluation
    print(f"\n{'='*55}")
    print(f"  Test Evaluation [{mode.upper()}]")
    print(f"{'='*55}")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    _, test_acc, test_auroc, preds, labels = evaluate(
        model, test_loader, device, loss_fn
    )
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test AUROC    : {test_auroc:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['Benign','Cancer'])}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen",   action="store_true",
                        help="Freeze encoder (linear probe)")
    parser.add_argument("--unfrozen", action="store_true",
                        help="Full fine-tuning (default)")
    args = parser.parse_args()

    freeze = args.frozen and not args.unfrozen
    train(freeze_encoder=freeze)
