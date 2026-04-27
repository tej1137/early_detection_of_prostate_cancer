"""
contrastive/train_fusion_contrastive.py

Fine-tuning contrastive pretrained MRI encoder + PSA clinical features
for multimodal cancer classification.

Architecture:
    Pretrained MRI Encoder  → (B, 512)
    PSA Encoder             → (B, 64)
    Concatenate             → (B, 576)
    Classification Head     → (B, 2)

Output:
    /workspace/checkpoints/contrastive/
        fusion_finetuned_frozen.pt
        fusion_finetuned_unfrozen.pt
        fusion_finetune_frozen_log.csv
        fusion_finetune_unfrozen_log.csv
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
from mri_baseline.models.mri_encoder       import MRIEncoder
from mri_baseline.models.psa_encoder       import PSAEncoder
from mri_baseline.data.multimodal_dataset  import PiCAIDataset


CFG = {
    "epochs"          : 50,
    "batch_size"      : 8,
    "lr_head"         : 1e-3,
    "lr_encoder"      : 1e-4,
    "weight_decay"    : 1e-4,
    "num_workers"     : 4,
    "mri_dim"         : 512,
    "clinical_dim"    : 64,

    "pretrained_path" : Path("/workspace/checkpoints/contrastive/best_encoder.pt"),
    "ckpt_dir"        : Path("/workspace/checkpoints/contrastive"),
}

# FUSION MODEL

class ContrastiveFusionModel(nn.Module):
    """
    Pretrained MRI encoder + PSA encoder + classification head.

    MRI   (B, 3, 20, 160, 160) → MRIEncoder    → (B, 512)
    PSA   (B, 4)               → PSAEncoder    → (B, 64)
    Concat                                      → (B, 576)
    Head                                        → (B, 2)
    """

    def __init__(self, freeze_encoder: bool = False):
        super().__init__()

        # MRI encoder — load pretrained weights
        self.mri_encoder = MRIEncoder(embedding_dim=CFG["mri_dim"])
        state = torch.load(CFG["pretrained_path"], map_location="cpu", weights_only=True)
        self.mri_encoder.load_state_dict(state)
        print(f"  ✓ Pretrained MRI encoder loaded")

        if freeze_encoder:
            for param in self.mri_encoder.parameters():
                param.requires_grad = False

        # PSA encoder — randomly initialised
        self.psa_encoder = PSAEncoder(embedding_dim=CFG["clinical_dim"])

        # Fusion head
        fusion_dim = CFG["mri_dim"] + CFG["clinical_dim"] 
        self.head  = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, mri: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        mri_feat = self.mri_encoder(mri)          # (B, 512)
        psa_feat = self.psa_encoder(clinical)     # (B, 64)
        fused    = torch.cat([mri_feat, psa_feat], dim=1)  # (B, 576)
        return self.head(fused)                   # (B, 2)


#Helpers

def get_loaders():
    train_ds = PiCAIDataset(split="train")
    val_ds   = PiCAIDataset(split="val")
    test_ds  = PiCAIDataset(split="test")

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
    acc      = np.mean(np.array(all_preds) == np.array(all_labels))
    auroc    = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auroc, all_preds, all_labels

#Training

def train(freeze_encoder: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode   = "frozen" if freeze_encoder else "unfrozen"

    print(f"\n{'='*55}")
    print(f"  Contrastive Fusion Fine-tuning  [{mode.upper()}]")
    print(f"{'='*55}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {CFG['epochs']}")
    print(f"  Batch size : {CFG['batch_size']}")
    print(f"  Mode       : {'Encoder FROZEN' if freeze_encoder else 'Full fine-tuning'}")
    print(f"{'='*55}\n")

    CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    model   = ContrastiveFusionModel(freeze_encoder=freeze_encoder).to(device)

    class_weights = torch.tensor([1.0, 2.0], device=device)
    loss_fn       = nn.CrossEntropyLoss(weight=class_weights)

    if freeze_encoder:
        optimizer = optim.AdamW([
            {"params": model.psa_encoder.parameters(), "lr": CFG["lr_head"]},
            {"params": model.head.parameters(),        "lr": CFG["lr_head"]},
        ], weight_decay=CFG["weight_decay"])
    else:
        optimizer = optim.AdamW([
            {"params": model.mri_encoder.parameters(), "lr": CFG["lr_encoder"]},
            {"params": model.psa_encoder.parameters(), "lr": CFG["lr_head"]},
            {"params": model.head.parameters(),        "lr": CFG["lr_head"]},
        ], weight_decay=CFG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6
    )
    scaler = GradScaler()

    train_loader, val_loader, test_loader = get_loaders()

    log_path = CFG["ckpt_dir"] / f"fusion_finetune_{mode}_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "val_acc", "val_auroc"
        ])

    best_auroc = 0.0
    ckpt_path  = CFG["ckpt_dir"] / f"fusion_finetuned_{mode}.pt"

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        if freeze_encoder:
            model.mri_encoder.eval()

        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            mri      = batch["mri"].to(device,     non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            labels   = batch["label"].to(device,    non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                logits = model(mri, clinical)
                loss   = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        avg_train  = train_loss / len(train_loader)
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
            print(f"  ✓ Best model saved (AUROC: {best_auroc:.3f})")

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

# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen",   action="store_true")
    parser.add_argument("--unfrozen", action="store_true")
    args = parser.parse_args()

    freeze = args.frozen and not args.unfrozen
    train(freeze_encoder=freeze)
