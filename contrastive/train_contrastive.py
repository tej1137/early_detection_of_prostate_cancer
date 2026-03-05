"""
contrastive/train_contrastive.py

SimCLR contrastive pretraining loop for MRI encoder.

What this script does:
    1. Loads all 1441 MRI scans (no labels)
    2. Trains SimCLR model using NTXent loss
    3. Saves the pretrained encoder weights for fine-tuning

Output:
    /workspace/checkpoints/contrastive/
        best_encoder.pt        ← best encoder weights (for fine-tuning)
        best_simclr.pt         ← full model checkpoint (to resume training)
        last_simclr.pt         ← final epoch checkpoint
        pretrain_log.csv       ← loss per epoch
"""

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import csv
import time
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from contrastive.contrastive_dataset import get_contrastive_loader
from contrastive.contrastive_model   import SimCLRModel, NTXentLoss


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

CFG = {
    # Training
    "epochs"       : 100,
    "batch_size"   : 16,
    "lr"           : 3e-4,
    "weight_decay" : 1e-4,
    "temperature"  : 0.07,
    "num_workers"  : 4,

    # Model
    "encoder_dim"  : 512,
    "proj_dim"     : 128,

    # Scheduler — cosine decay to near zero
    "lr_min"       : 1e-6,

    # Paths
    "ckpt_dir"     : Path("/workspace/checkpoints/contrastive"),
}


# ══════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  SimCLR Contrastive Pretraining")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs      : {CFG['epochs']}")
    print(f"  Batch size  : {CFG['batch_size']}")
    print(f"  LR          : {CFG['lr']}")
    print(f"  Temperature : {CFG['temperature']}")
    print(f"{'='*55}\n")

    # ── Setup ──────────────────────────────────────────────
    CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    loader  = get_contrastive_loader(
        batch_size  = CFG["batch_size"],
        num_workers = CFG["num_workers"],
    )

    model   = SimCLRModel(
        encoder_out_dim = CFG["encoder_dim"],
        proj_out_dim    = CFG["proj_dim"],
    ).to(device)

    loss_fn = NTXentLoss(temperature=CFG["temperature"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    # Cosine annealing — smoothly decays LR over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = CFG["epochs"],
        eta_min= CFG["lr_min"],
    )

    scaler = GradScaler()   # mixed precision

    # ── Log file ───────────────────────────────────────────
    log_path = CFG["ckpt_dir"] / "pretrain_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "lr", "epoch_time_s"])

    # ── Train ──────────────────────────────────────────────
    best_loss = float("inf")

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (view1, view2) in enumerate(loader):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                z1   = model(view1)          # (B, 128)
                z2   = model(view2)          # (B, 128)
                loss = loss_fn(z1, z2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss   = epoch_loss / len(loader)
        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # ── Logging ────────────────────────────────────────
        print(f"  Epoch [{epoch:>3}/{CFG['epochs']}]  "
              f"Loss: {avg_loss:.4f}  "
              f"LR: {current_lr:.2e}  "
              f"Time: {epoch_time:.1f}s")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(avg_loss, 6),
                             round(current_lr, 8), round(epoch_time, 1)])

        # ── Checkpointing ──────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss

            # Encoder weights only — for fine-tuning
            torch.save(
                model.get_encoder_weights(),
                CFG["ckpt_dir"] / "best_encoder.pt"
            )

            # Full model — to resume pretraining if needed
            torch.save({
                "epoch"     : epoch,
                "model"     : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "loss"      : best_loss,
            }, CFG["ckpt_dir"] / "best_simclr.pt")

            print(f"  ✓ Best model saved (loss: {best_loss:.4f})")

    # ── Final checkpoint ───────────────────────────────────
    torch.save({
        "epoch"     : CFG["epochs"],
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "loss"      : avg_loss,
    }, CFG["ckpt_dir"] / "last_simclr.pt")

    print(f"\n{'='*55}")
    print(f"  Pretraining complete!")
    print(f"  Best loss     : {best_loss:.4f}")
    print(f"  Encoder saved : {CFG['ckpt_dir']}/best_encoder.pt")
    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
