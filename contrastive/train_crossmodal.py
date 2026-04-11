"""
contrastive/train_crossmodal.py

Cross-Modal Contrastive Pretraining Loop.

What this script does:
    1. Loads ALL 1441 patients (no labels — self-supervised)
    2. For each patient: loads MRI + clinical features
    3. Trains CrossModalModel with symmetric InfoNCE loss
       → MRI encoder learns to match its patient's clinical profile
       → Clinical encoder learns to match its patient's MRI
    4. Saves BOTH pretrained encoder weights for fine-tuning

This is fundamentally different from train_contrastive.py (SimCLR):
    SimCLR:      one encoder, two views of the SAME modality (MRI aug1 vs aug2)
    Cross-modal: two encoders, two DIFFERENT modalities (MRI vs clinical)

Output:
    /workspace/checkpoints/crossmodal/
        best_mri_encoder.pt         ← best MRI encoder weights (for fine-tuning)
        best_clinical_encoder.pt    ← best clinical encoder weights (for fine-tuning)
        best_crossmodal.pt          ← full model checkpoint (to resume)
        last_crossmodal.pt          ← final epoch checkpoint
        pretrain_log.csv            ← loss per epoch

Run:
    python -m contrastive.train_crossmodal
"""

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import csv
import time
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from contrastive.crossmodal_dataset import get_crossmodal_loader
from contrastive.crossmodal_model   import CrossModalModel, InfoNCELoss


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

CFG = {
    # Training
    "epochs"             : 100,
    "batch_size"         : 32,      # RTX 5090 32GB — larger batch = more negatives
    "lr"                 : 3e-4,
    "weight_decay"       : 1e-4,
    "num_workers"        : 4,

    # Model
    "mri_encoder_dim"    : 512,
    "clinical_encoder_dim": 128,
    "proj_out_dim"       : 128,     # shared embedding space

    # Initial temperature — model will learn to adjust it
    "temperature"        : 0.07,

    # Scheduler — cosine decay
    "lr_min"             : 1e-6,

    # Gradient clipping
    "max_grad_norm"      : 1.0,

    # Paths
    "ckpt_dir"           : Path("/workspace/checkpoints/crossmodal"),
}


# ══════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Cross-Modal Contrastive Pretraining")
    print(f"  MRI ↔ Clinical  (symmetric InfoNCE / CLIP-style)")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM        : {mem:.1f} GB")
    print(f"  Epochs      : {CFG['epochs']}")
    print(f"  Batch size  : {CFG['batch_size']}")
    print(f"  LR          : {CFG['lr']}")
    print(f"  Proj dim    : {CFG['proj_out_dim']}")
    print(f"{'='*60}\n")

    # ── Setup ──────────────────────────────────────────────
    CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    loader = get_crossmodal_loader(
        batch_size  = CFG["batch_size"],
        num_workers = CFG["num_workers"],
        splits      = ["train", "val", "test"],   # ALL data — no labels used
        augment     = True,
    )

    model = CrossModalModel(
        mri_encoder_dim      = CFG["mri_encoder_dim"],
        clinical_encoder_dim = CFG["clinical_encoder_dim"],
        proj_out_dim         = CFG["proj_out_dim"],
        temperature          = CFG["temperature"],
    ).to(device)

    loss_fn = InfoNCELoss()

    # Temperature is a model parameter — include in optimiser
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = CFG["epochs"],
        eta_min = CFG["lr_min"],
    )

    scaler = GradScaler()   # mixed precision — important for RTX 5090

    # ── Log file ───────────────────────────────────────────
    log_path = CFG["ckpt_dir"] / "pretrain_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "loss", "loss_mri2clin", "loss_clin2mri",
            "temperature", "lr", "epoch_time_s"
        ])

    # ── Print model sizes ──────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    mri_params   = sum(p.numel() for p in model.mri_encoder.parameters())
    clin_params  = sum(p.numel() for p in model.clinical_encoder.parameters())
    print(f"  Model parameters:")
    print(f"    MRI encoder    : {mri_params:>10,}")
    print(f"    Clinical encoder: {clin_params:>9,}")
    print(f"    Total (inc. heads): {total_params:>6,}")
    print(f"\n  Batches per epoch: {len(loader)}")
    print(f"  Negatives per sample: {CFG['batch_size'] - 1}\n")

    # ── Train ──────────────────────────────────────────────
    best_loss = float("inf")

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        epoch_loss      = 0.0
        epoch_loss_mri  = 0.0
        epoch_loss_clin = 0.0
        t0 = time.time()

        for batch in loader:
            mri      = batch["mri"].to(device,      non_blocking=True)
            clinical = batch["clinical"].to(device,  non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                z_mri, z_clin = model(mri, clinical)
                loss, loss_mri, loss_clin = loss_fn(
                    z_mri, z_clin, model.temperature
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=CFG["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()

            epoch_loss      += loss.item()
            epoch_loss_mri  += loss_mri.item()
            epoch_loss_clin += loss_clin.item()

        scheduler.step()

        # ── Epoch stats ────────────────────────────────────
        n_batches  = len(loader)
        avg_loss   = epoch_loss      / n_batches
        avg_mri    = epoch_loss_mri  / n_batches
        avg_clin   = epoch_loss_clin / n_batches
        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        temp_val   = model.temperature.item()

        print(
            f"  Epoch [{epoch:>3}/{CFG['epochs']}]  "
            f"Loss: {avg_loss:.4f}  "
            f"(MRI→Clin: {avg_mri:.4f}  Clin→MRI: {avg_clin:.4f})  "
            f"Temp: {temp_val:.4f}  "
            f"LR: {current_lr:.2e}  "
            f"Time: {epoch_time:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(avg_loss,   6),
                round(avg_mri,    6),
                round(avg_clin,   6),
                round(temp_val,   6),
                round(current_lr, 8),
                round(epoch_time, 1),
            ])

        # ── Checkpointing ──────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss

            # Save MRI encoder weights only — for fusion fine-tuning
            torch.save(
                model.get_mri_encoder_weights(),
                CFG["ckpt_dir"] / "best_mri_encoder.pt"
            )

            # Save clinical encoder weights only
            torch.save(
                model.get_clinical_encoder_weights(),
                CFG["ckpt_dir"] / "best_clinical_encoder.pt"
            )

            # Save full model — to resume pretraining
            torch.save({
                "epoch"      : epoch,
                "model"      : model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "loss"       : best_loss,
                "temperature": temp_val,
            }, CFG["ckpt_dir"] / "best_crossmodal.pt")

            print(f"  ✓ Best model saved  (loss: {best_loss:.4f})")

    # ── Final checkpoint ───────────────────────────────────
    torch.save({
        "epoch"      : CFG["epochs"],
        "model"      : model.state_dict(),
        "optimizer"  : optimizer.state_dict(),
        "loss"       : avg_loss,
        "temperature": model.temperature.item(),
    }, CFG["ckpt_dir"] / "last_crossmodal.pt")

    print(f"\n{'='*60}")
    print(f"  Pretraining complete!")
    print(f"  Best loss            : {best_loss:.4f}")
    print(f"  Final temperature    : {model.temperature.item():.4f}")
    print(f"  MRI encoder saved    : {CFG['ckpt_dir']}/best_mri_encoder.pt")
    print(f"  Clinical enc. saved  : {CFG['ckpt_dir']}/best_clinical_encoder.pt")
    print(f"{'='*60}\n")
    print(f"  Next step: run train_fusion_crossmodal.py")


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()