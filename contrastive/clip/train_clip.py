"""
contrastive/train_clip.py

CLIP-Style Pretraining Loop — handles both Option A and Option B.

Option A (--option_a):
    Both encoders randomly initialised.
    Learns MRI↔clinical alignment purely from scratch.
    Cleanest comparison point vs SimCLR and cross-modal.

Option B (--option_b):
    MRI encoder warm-started from SimCLR best_encoder.pt.
    Clinical encoder randomly initialised.
    Tests whether SimCLR pretraining + CLIP alignment is additive.

Key differences vs train_crossmodal.py:
    1. Wider clinical encoder (256-dim vs 128-dim)
    2. Deeper projection heads (3-layer vs 2-layer)
    3. Separate LR groups: MRI encoder trains at 1e-4, clinical at 3e-4
       (MRI encoder is larger and more sensitive — needs lower LR)
    4. Logs positive/negative pair similarities per epoch
       (lets you see how well alignment is progressing)
    5. Option B: MRI encoder warm-started from SimCLR

Output:
    /workspace/checkpoints/clip_a/   (Option A)
        best_mri_encoder.pt
        best_clinical_encoder.pt
        best_clip.pt
        last_clip.pt
        pretrain_log.csv

    /workspace/checkpoints/clip_b/   (Option B)
        (same structure)

Run:
    python -m contrastive.train_clip --option_a
    python -m contrastive.train_clip --option_b
    python -m contrastive.train_clip --option_b --freeze_mri
"""

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import csv
import time
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from contrastive.crossmodal_dataset import get_crossmodal_loader  # lives in contrastive/, not contrastive/clip/
from contrastive.clip.clip_model    import CLIPModel, CLIPModelB, CLIPLoss


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

CFG = {
    # Training
    "epochs"              : 100,
    "batch_size"          : 32,       # RTX 5090 32GB — more negatives = better
    "num_workers"         : 4,

    # Separate LRs for MRI vs clinical encoder
    # MRI encoder is larger + more sensitive → lower LR
    # Clinical encoder is small + random    → higher LR
    "lr_mri_encoder"      : 1e-4,
    "lr_clinical_encoder" : 3e-4,
    "lr_projectors"       : 3e-4,
    "lr_temperature"      : 1e-3,     # temperature adapts fastest
    "weight_decay"        : 1e-4,

    # Model dims
    "mri_encoder_dim"     : 512,
    "clinical_encoder_dim": 256,      # wider than crossmodal (128)
    "proj_out_dim"        : 128,

    # Temperature
    "init_temperature"    : 0.07,

    # Scheduler
    "lr_min"              : 1e-6,

    # Gradient clipping
    "max_grad_norm"       : 1.0,

    # SimCLR checkpoint for Option B
    "simclr_ckpt"         : Path("/workspace/checkpoints/contrastive/best_encoder.pt"),
}


# ══════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════

def train(option: str, freeze_mri: bool = False):
    """
    Args:
        option    : "A" or "B"
        freeze_mri: (Option B only) freeze MRI encoder during CLIP pretraining
    """
    assert option in ("A", "B"), "option must be 'A' or 'B'"

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(f"/workspace/checkpoints/clip_{option.lower()}")

    # Descriptive label for logs
    if option == "B" and freeze_mri:
        label = "Option B (SimCLR warm-start, MRI FROZEN)"
    elif option == "B":
        label = "Option B (SimCLR warm-start, full CLIP)"
    else:
        label = "Option A (pure CLIP, from scratch)"

    print(f"\n{'='*60}")
    print(f"  CLIP-Style Pretraining — {label}")
    print(f"{'='*60}")
    print(f"  Device           : {device}")
    if torch.cuda.is_available():
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM             : {mem:.1f} GB")
    print(f"  Epochs           : {CFG['epochs']}")
    print(f"  Batch size       : {CFG['batch_size']}")
    print(f"  Clinical enc dim : {CFG['clinical_encoder_dim']} (wider than crossmodal)")
    print(f"  Proj dim         : {CFG['proj_out_dim']}")
    print(f"  Negatives/sample : {CFG['batch_size'] - 1}")
    print(f"{'='*60}\n")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ──────────────────────────────────────────────
    if option == "A":
        model = CLIPModel(
            mri_encoder_dim      = CFG["mri_encoder_dim"],
            clinical_encoder_dim = CFG["clinical_encoder_dim"],
            proj_out_dim         = CFG["proj_out_dim"],
            init_temperature     = CFG["init_temperature"],
        ).to(device)
    else:
        model = CLIPModelB(
            simclr_ckpt          = CFG["simclr_ckpt"],
            freeze_mri           = freeze_mri,
            mri_encoder_dim      = CFG["mri_encoder_dim"],
            clinical_encoder_dim = CFG["clinical_encoder_dim"],
            proj_out_dim         = CFG["proj_out_dim"],
            init_temperature     = CFG["init_temperature"],
        ).to(device)

    loss_fn = CLIPLoss()

    # ── Separate param groups — different LRs ─────────────
    # This is the key difference vs train_crossmodal.py which
    # uses a single LR for everything
    param_groups = [
        {
            "params" : model.clinical_encoder.parameters(),
            "lr"     : CFG["lr_clinical_encoder"],
            "name"   : "clinical_encoder",
        },
        {
            "params" : model.mri_projector.parameters(),
            "lr"     : CFG["lr_projectors"],
            "name"   : "mri_projector",
        },
        {
            "params" : model.clinical_projector.parameters(),
            "lr"     : CFG["lr_projectors"],
            "name"   : "clinical_projector",
        },
        {
            "params" : [model.log_temperature],
            "lr"     : CFG["lr_temperature"],
            "name"   : "temperature",
        },
    ]

    # Add MRI encoder only if not frozen
    if not (option == "B" and freeze_mri):
        param_groups.insert(0, {
            "params" : model.mri_encoder.parameters(),
            "lr"     : CFG["lr_mri_encoder"],   # lower LR for MRI encoder
            "name"   : "mri_encoder",
        })

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=CFG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = CFG["epochs"],
        eta_min = CFG["lr_min"],
    )

    scaler = GradScaler()

    # ── Dataloader ─────────────────────────────────────────
    loader = get_crossmodal_loader(
        batch_size  = CFG["batch_size"],
        num_workers = CFG["num_workers"],
        splits      = ["train", "val", "test"],
        augment     = True,
    )

    # ── Print model summary ────────────────────────────────
    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters : {total_params:,}  ({trainable:,} trainable)")
    print(f"  Batches/epoch    : {len(loader)}\n")

    # ── Log file ───────────────────────────────────────────
    log_path = ckpt_dir / "pretrain_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "loss", "loss_mri2clin", "loss_clin2mri",
            "pos_sim", "neg_sim", "temperature", "lr_mri", "epoch_time_s",
        ])

    # ── Training ───────────────────────────────────────────
    best_loss = float("inf")

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        if option == "B" and freeze_mri:
            model.mri_encoder.eval()

        epoch_loss     = 0.0
        epoch_loss_mri = 0.0
        epoch_loss_clin= 0.0
        epoch_pos_sim  = 0.0
        epoch_neg_sim  = 0.0
        t0             = time.time()

        for batch in loader:
            mri      = batch["mri"].to(device,     non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                z_mri, z_clin = model(mri, clinical)
                loss, loss_mri, loss_clin, pos_sim, neg_sim = loss_fn(
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
            epoch_pos_sim   += pos_sim.item()
            epoch_neg_sim   += neg_sim.item()

        scheduler.step()

        # ── Epoch stats ────────────────────────────────────
        n          = len(loader)
        avg_loss   = epoch_loss      / n
        avg_mri    = epoch_loss_mri  / n
        avg_clin   = epoch_loss_clin / n
        avg_pos    = epoch_pos_sim   / n
        avg_neg    = epoch_neg_sim   / n
        temp_val   = model.temperature.item()
        epoch_time = time.time() - t0

        # Get MRI encoder LR (first group if not frozen, else N/A)
        if not (option == "B" and freeze_mri):
            lr_mri = optimizer.param_groups[0]["lr"]
        else:
            lr_mri = 0.0

        print(
            f"  Epoch [{epoch:>3}/{CFG['epochs']}]  "
            f"Loss: {avg_loss:.4f}  "
            f"(M→C: {avg_mri:.4f}  C→M: {avg_clin:.4f})  "
            f"Sim pos: {avg_pos:.3f}  neg: {avg_neg:.3f}  "
            f"Temp: {temp_val:.4f}  "
            f"Time: {epoch_time:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                round(avg_loss,  6), round(avg_mri,  6), round(avg_clin, 6),
                round(avg_pos,   4), round(avg_neg,   4),
                round(temp_val,  6), round(lr_mri,    8),
                round(epoch_time,1),
            ])

        # ── Checkpoint ─────────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save(
                model.get_mri_encoder_weights(),
                ckpt_dir / "best_mri_encoder.pt"
            )
            torch.save(
                model.get_clinical_encoder_weights(),
                ckpt_dir / "best_clinical_encoder.pt"
            )
            torch.save({
                "epoch"      : epoch,
                "option"     : option,
                "freeze_mri" : freeze_mri,
                "model"      : model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "loss"       : best_loss,
                "temperature": temp_val,
            }, ckpt_dir / "best_clip.pt")

            print(f"  ✓ Best model saved  (loss: {best_loss:.4f})")

    # ── Final checkpoint ───────────────────────────────────
    torch.save({
        "epoch"      : CFG["epochs"],
        "option"     : option,
        "model"      : model.state_dict(),
        "loss"       : avg_loss,
        "temperature": model.temperature.item(),
    }, ckpt_dir / "last_clip.pt")

    print(f"\n{'='*60}")
    print(f"  CLIP Pretraining complete! [{label}]")
    print(f"  Best loss            : {best_loss:.4f}")
    print(f"  Final temperature    : {model.temperature.item():.4f}")
    print(f"  MRI encoder saved    : {ckpt_dir}/best_mri_encoder.pt")
    print(f"  Clinical enc. saved  : {ckpt_dir}/best_clinical_encoder.pt")
    print(f"{'='*60}")
    print(f"\n  Next: run train_fusion_clip.py --option_{option.lower()}")


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP-style cross-modal pretraining for prostate cancer"
    )
    parser.add_argument(
        "--option_a", action="store_true",
        help="Pure CLIP from scratch (Option A)"
    )
    parser.add_argument(
        "--option_b", action="store_true",
        help="CLIP with SimCLR warm-started MRI encoder (Option B)"
    )
    parser.add_argument(
        "--freeze_mri", action="store_true",
        help="(Option B only) Freeze MRI encoder during CLIP pretraining"
    )
    args = parser.parse_args()

    if args.option_a and args.option_b:
        print("  ERROR: specify only one of --option_a or --option_b")
        exit(1)

    if not args.option_a and not args.option_b:
        print("  No option specified — defaulting to --option_a")
        train(option="A")
    elif args.option_a:
        train(option="A")
    else:
        train(option="B", freeze_mri=args.freeze_mri)