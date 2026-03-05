"""
preprocessing/preprocess_mri.py

ONE-TIME preprocessing script for PI-CAI MRI data.

What this does:
  For each case in clinical_preprocessed.csv:
    1. Finds T2W, ADC, HBV .mha files on disk
    2. Resamples to consistent voxel spacing
    3. Crops/pads to fixed volume size
    4. Normalises intensities per-sequence
    5. Saves as 3-channel .pt tensor → (3, D, H, W)

Run once (on RunPod — do NOT run locally, no MRI data):
  python -m mri_baseline.preprocessing.preprocess_mri

Outputs:
  /workspace/data/preprocessed/mri/{case_id}.pt   ← one file per case
  /workspace/data/preprocessed/missing_mri_cases.txt
  /workspace/data/preprocessed/mri_audit.json
"""

import os
import json
import time
import traceback
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from pathlib import Path
from tqdm import tqdm


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

class Config:
    # ── Input ──────────────────────────────────────────────
    clinical_csv  = Path("/workspace/data/preprocessed/clinical_preprocessed.csv")
    mri_root      = Path("/workspace/data/images")   # ← parent of all folds
    mri_folds     = ["picai_public_images_fold0",
                     "picai_public_images_fold1",
                     "picai_public_images_fold2",
                     "picai_public_images_fold3",
                     "picai_public_images_fold4"]    # ← search all folds

    # ── Output ─────────────────────────────────────────────
    output_dir    = Path("/workspace/data/preprocessed/mri")
    missing_log   = Path("/workspace/data/preprocessed/missing_mri_cases.txt")
    audit_json    = Path("/workspace/data/preprocessed/mri_audit.json")

    # ── MRI sequences to load ──────────────────────────────
    # These 3 channels stack into (3, D, H, W)
    sequences     = ["t2w", "adc", "hbv"]

    # ── Resampling ─────────────────────────────────────────
    # Target voxel spacing in mm: (axial_slice_thickness, in-plane, in-plane)
    # PI-CAI T2W is typically ~3mm slice thickness, 0.5mm in-plane
    # We resample everything to uniform spacing so all volumes are comparable
    target_spacing = (3.0, 0.5, 0.5)   # (z, y, x) in mm

    # ── Volume size after crop/pad ─────────────────────────
    # Fixed spatial dimensions: (depth, height, width)
    # After resampling to 3mm z-spacing: ~20 slices covers the prostate
    # After resampling to 0.5mm xy: 160x160 covers the prostate ROI
    target_size    = (20, 160, 160)     # (D, H, W)

    # ── Intensity normalisation ────────────────────────────
    # We clip to percentiles first to remove extreme outliers,
    # then normalise to [0, 1] range.
    clip_percentile_low  = 1.0    # clip values below 1st percentile
    clip_percentile_high = 99.0   # clip values above 99th percentile

    # ── Skip already processed cases ──────────────────────
    # If True: skip cases where .pt already exists (for resuming)
    # If False: reprocess everything from scratch
    skip_existing  = True

    def __init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# STEP 1 — AUDIT: FIND MISSING MRI FILES
# ══════════════════════════════════════════════════════════

def find_case_dir(case_id: str, config: Config):
    """
    Search all fold directories for a case.
    Structure: {mri_root}/{fold}/{patient_id}/{case_id}_{seq}.mha
    Returns the fold directory Path if found, else None.
    """
    patient_id = case_id.split("_")[0]
    for fold in config.mri_folds:
        candidate = config.mri_root / fold / patient_id
        if candidate.exists():
            return candidate
    return None


def audit_mri_files(config: Config, case_ids: list) -> tuple[list, list]:
    print("\n" + "═"*60)
    print("STEP 1 — AUDITING MRI FILES ON DISK")
    print("═"*60)

    valid_cases    = []
    missing_cases  = []
    missing_detail = {}

    for case_id in tqdm(case_ids, desc="  Auditing"):
        case_dir = find_case_dir(case_id, config)

        if case_dir is None:
            missing_cases.append(case_id)
            missing_detail[case_id] = ["patient_dir_not_found"]
            continue

        missing_seqs = []
        for seq in config.sequences:
            fpath = case_dir / f"{case_id}_{seq}.mha"
            if not fpath.exists():
                missing_seqs.append(seq)

        if missing_seqs:
            missing_cases.append(case_id)
            missing_detail[case_id] = missing_seqs
        else:
            valid_cases.append(case_id)

    print(f"\n  Total cases   : {len(case_ids)}")
    print(f"  Valid cases   : {len(valid_cases)}  ← all 3 sequences present")
    print(f"  Missing cases : {len(missing_cases)}  ← at least 1 sequence missing")

    if missing_cases:
        print(f"\n  Missing breakdown:")
        for cid, seqs in missing_detail.items():
            print(f"    {cid}  missing: {seqs}")

        with open(config.missing_log, 'w') as f:
            f.write("# Cases missing MRI files\n")
            for cid in missing_cases:
                f.write(f"{cid}  # missing: {missing_detail[cid]}\n")
        print(f"\n  ✓ Missing cases logged to: {config.missing_log}")

    return valid_cases, missing_cases   # ← THIS LINE WAS MISSING


# ══════════════════════════════════════════════════════════
# STEP 2 — LOAD MRI VOLUME
# ══════════════════════════════════════════════════════════

def load_mha(fpath: Path) -> sitk.Image:
    """
    Load a .mha file as a SimpleITK image.

    SimpleITK preserves:
      - Voxel spacing (mm per voxel)
      - Image origin (world coordinates)
      - Direction cosines (orientation)

    We need spacing to resample correctly.
    """
    return sitk.ReadImage(str(fpath), sitk.sitkFloat32)


# ══════════════════════════════════════════════════════════
# STEP 3 — RESAMPLE TO UNIFORM SPACING
# ══════════════════════════════════════════════════════════

def resample_volume(image: sitk.Image, target_spacing: tuple) -> sitk.Image:
    """
    Resample image to a fixed voxel spacing using B-spline interpolation.

    WHY we resample:
      Different scanners acquire at different voxel spacings.
      e.g. one case: spacing = (3.0, 0.45, 0.45)
           another:  spacing = (3.6, 0.52, 0.52)
      After resampling, all cases have spacing = (3.0, 0.5, 0.5).
      The model then sees the same physical scale for every case.

    Interpolation:
      B-spline (order 3) is used for MRI intensity volumes.
      It's smoother than linear but doesn't introduce ringing artifacts
      like higher-order methods. For binary masks, use NearestNeighbor.

    Steps:
      1. Compute new output size from old_size × (old_spacing / new_spacing)
      2. Set up the resampler with the new spacing + size
      3. Apply to the image
    """
    original_spacing = image.GetSpacing()          # (x, y, z) in mm
    original_size    = image.GetSize()             # (x, y, z) in voxels

    # SimpleITK uses (x, y, z) ordering — our config uses (z, y, x)
    # so we reverse target_spacing for SimpleITK
    sitk_target_spacing = (
        target_spacing[2],   # x
        target_spacing[1],   # y
        target_spacing[0],   # z
    )

    # Compute new size so physical extent is preserved
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / sitk_target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(sitk_target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(image)


# ══════════════════════════════════════════════════════════
# STEP 4 — CROP OR PAD TO FIXED SIZE
# ══════════════════════════════════════════════════════════

def crop_or_pad(volume: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize a 3D numpy array to (D, H, W) by cropping or zero-padding.

    volume shape: (D, H, W)    — current, variable
    target_size:  (D, H, W)    — fixed output size

    Strategy: centre-crop if too large, centre-pad if too small.

    WHY centre:
      The prostate sits roughly in the centre of the pelvic MRI volume.
      Centre-cropping removes peripheral anatomy (legs, bladder wall).
      Centre-padding adds symmetric black borders on both sides.
      This keeps the prostate near the centre of every output volume.
    """
    result = np.zeros(target_size, dtype=np.float32)

    for dim in range(3):
        curr = volume.shape[dim]
        tgt  = target_size[dim]

        if curr >= tgt:
            # Crop: take centre tgt voxels
            start = (curr - tgt) // 2
            volume = np.take(volume, range(start, start + tgt), axis=dim)
        else:
            # Pad: insert into centre of zero array
            pass   # handled by result initialisation below

    # Now insert cropped volume into centre of zero-padded result
    slices = []
    for dim in range(3):
        curr = volume.shape[dim]
        tgt  = target_size[dim]
        if curr < tgt:
            start = (tgt - curr) // 2
            slices.append(slice(start, start + curr))
        else:
            slices.append(slice(0, tgt))

    result[tuple(slices)] = volume
    return result


# ══════════════════════════════════════════════════════════
# STEP 5 — NORMALISE INTENSITY
# ══════════════════════════════════════════════════════════

def normalise_intensity(volume: np.ndarray, clip_low: float, clip_high: float) -> np.ndarray:
    """
    Normalise MRI intensity values to [0, 1].

    Step 1 — Percentile clipping:
      MRI volumes often have extreme outlier voxels (scanner artefacts,
      bright fat signal, air outside body). We clip at the 1st and 99th
      percentile to remove these before normalising.

    Step 2 — Min-max normalisation to [0, 1]:
      After clipping, stretch the remaining range to [0, 1].
      Each sequence (T2W, ADC, HBV) is normalised independently
      because they have completely different intensity scales:
        T2W:  0–1500 (arbitrary scanner units)
        ADC:  0–3000 (×10⁻⁶ mm²/s)
        HBV:  0–800  (arbitrary scanner units)

    Note: We do NOT use global mean/std normalisation here because:
      - MRI intensity is not standardised across scanners
      - Z-score normalisation can go negative (bad for some activations)
      - Per-case [0,1] is standard in medical imaging deep learning
    """
    # Compute percentiles on non-zero voxels only
    # (zero voxels are background/padding — shouldn't affect the scale)
    nonzero = volume[volume > 0]
    if len(nonzero) == 0:
        return volume  # all-zero volume — return as is

    p_low  = np.percentile(nonzero, clip_low)
    p_high = np.percentile(nonzero, clip_high)

    # Clip
    volume = np.clip(volume, p_low, p_high)

    # Min-max normalise
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 1e-8:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume.astype(np.float32)


# ══════════════════════════════════════════════════════════
# STEP 6 — PROCESS ONE CASE
# ══════════════════════════════════════════════════════════

def process_case(case_id: str, config: Config) -> bool:
    output_path = config.output_dir / f"{case_id}.pt"
    if config.skip_existing and output_path.exists():
        return True

    case_dir = find_case_dir(case_id, config)   # ← use find_case_dir
    if case_dir is None:
        raise FileNotFoundError(f"No MRI directory found for {case_id}")

    channels = []
    for seq in config.sequences:
        fpath = case_dir / f"{case_id}_{seq}.mha"
        image = load_mha(fpath)
        image = resample_volume(image, config.target_spacing)
        volume = sitk.GetArrayFromImage(image)
        volume = crop_or_pad(volume, config.target_size)
        volume = normalise_intensity(volume, config.clip_percentile_low, config.clip_percentile_high)
        channels.append(volume)

    tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)
    assert tensor.shape == (3,) + config.target_size
    torch.save(tensor, output_path)
    return True


# ══════════════════════════════════════════════════════════
# STEP 7 — PROCESS ALL CASES
# ══════════════════════════════════════════════════════════

def process_all_cases(valid_cases: list, config: Config) -> dict:
    """
    Run process_case() for every valid case with a progress bar.

    Failures are caught per-case so one corrupt file doesn't
    stop the entire preprocessing run.

    Returns audit dict with success/failure counts and failed case list.
    """
    print("\n" + "═"*60)
    print("STEP 2 — PREPROCESSING MRI VOLUMES")
    print("═"*60)
    print(f"\n  Cases to process : {len(valid_cases)}")
    print(f"  Output shape     : (3, {config.target_size[0]}, {config.target_size[1]}, {config.target_size[2]})")
    print(f"  Output dir       : {config.output_dir}")
    print(f"  Skip existing    : {config.skip_existing}")

    successes  = []
    failures   = {}
    skipped    = 0

    start_time = time.time()

    for case_id in tqdm(valid_cases, desc="  Processing"):
        # Check if skipped
        if config.skip_existing and (config.output_dir / f"{case_id}.pt").exists():
            skipped += 1
            successes.append(case_id)
            continue

        try:
            process_case(case_id, config)
            successes.append(case_id)
        except Exception as e:
            failures[case_id] = traceback.format_exc()

    elapsed = time.time() - start_time

    print(f"\n  ✓ Processed successfully : {len(successes) - skipped}")
    print(f"  ⏭  Skipped (already done): {skipped}")
    print(f"  ❌ Failed                : {len(failures)}")
    print(f"  ⏱  Time elapsed          : {elapsed/60:.1f} min")

    if failures:
        print(f"\n  Failed cases:")
        for cid, err in failures.items():
            print(f"    {cid}: {err.splitlines()[-1]}")

    return {
        "total"    : len(valid_cases),
        "success"  : len(successes) - skipped,
        "skipped"  : skipped,
        "failed"   : len(failures),
        "failed_cases" : list(failures.keys()),
        "elapsed_min"  : round(elapsed / 60, 2)
    }


# ══════════════════════════════════════════════════════════
# STEP 8 — VALIDATE OUTPUTS
# ══════════════════════════════════════════════════════════

def validate_outputs(valid_cases: list, config: Config, audit: dict):
    """
    Post-processing validation:
      1. Every expected .pt file exists
      2. Spot-check 5 random .pt files:
           - Correct shape: (3, D, H, W)
           - Values in [0, 1]
           - No NaN or Inf
      3. Save audit JSON with all stats
    """
    print("\n" + "═"*60)
    print("STEP 3 — VALIDATING OUTPUTS")
    print("═"*60)

    errors = []

    # Check 1: All .pt files exist
    missing_pt = [cid for cid in valid_cases
                  if not (config.output_dir / f"{cid}.pt").exists()]
    if missing_pt:
        errors.append(f"❌ {len(missing_pt)} .pt files not found: {missing_pt[:5]}")
    else:
        print(f"  ✓ All {len(valid_cases)} .pt files present")

    # Check 2: Spot-check 5 random cases
    import random
    sample = random.sample(valid_cases, min(5, len(valid_cases)))
    print(f"\n  Spot-checking {len(sample)} random cases:")

    for cid in sample:
        t = torch.load(config.output_dir / f"{cid}.pt", weights_only=True)
        expected_shape = (3,) + config.target_size

        shape_ok  = t.shape == expected_shape
        range_ok  = (t.min() >= 0.0) and (t.max() <= 1.0)
        finite_ok = torch.isfinite(t).all().item()

        status = "✓" if (shape_ok and range_ok and finite_ok) else "❌"
        print(f"    {status}  {cid}  shape={tuple(t.shape)}  "
              f"range=[{t.min():.3f}, {t.max():.3f}]  finite={finite_ok}")

        if not shape_ok:
            errors.append(f"❌ {cid}: shape {t.shape} != {expected_shape}")
        if not range_ok:
            errors.append(f"❌ {cid}: values outside [0,1]: [{t.min():.3f}, {t.max():.3f}]")
        if not finite_ok:
            errors.append(f"❌ {cid}: NaN or Inf values found")

    # Save audit JSON
    audit["validation_errors"] = errors
    audit["spot_checked"] = sample
    with open(config.audit_json, 'w') as f:
        json.dump(audit, f, indent=2)
    print(f"\n  ✓ Audit saved: {config.audit_json}")

    if errors:
        print("\n  VALIDATION ERRORS:")
        for e in errors:
            print(f"    {e}")
    else:
        print(f"\n  ✓ All validation checks passed")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("MRI PREPROCESSING")
    print("PI-CAI .mha files → Preprocessed .pt tensors")
    print("=" * 60)

    config = Config()

    # Load case IDs from clinical CSV
    df = pd.read_csv(config.clinical_csv, index_col="case_id")
    case_ids = df.index.tolist()
    print(f"\n  Cases in clinical CSV: {len(case_ids)}")

    # Step 1: Audit — find all missing MRI files
    valid_cases, missing_cases = audit_mri_files(config, case_ids)

    if not valid_cases:
        raise RuntimeError("No valid cases found — check mri_root path in Config")

    if missing_cases:
        print(f"\n  ⚠️  {len(missing_cases)} cases have missing MRI files.")
        print(f"  → These are logged to: {config.missing_log}")
        print(f"  → Add them to preprocess_clinical.py and re-run it first.")
        print(f"  → Continuing to preprocess the {len(valid_cases)} valid cases...\n")

    # Step 2: Process all valid cases
    audit = process_all_cases(valid_cases, config)

    # Step 3: Validate outputs
    validate_outputs(valid_cases, config, audit)

    print("\n" + "=" * 60)
    print("MRI PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n  Preprocessed .pt files : {config.output_dir}")
    print(f"  Missing cases log      : {config.missing_log}")
    print(f"  Audit report           : {config.audit_json}")

    if missing_cases:
        print(f"\n  ⚠️  NEXT STEP BEFORE TRAINING:")
        print(f"  1. Open preprocess_clinical.py")
        print(f"  2. Add cases from {config.missing_log} to Config.known_missing_mri")
        print(f"  3. Re-run preprocess_clinical.py")
        print(f"  4. Then begin training")
    else:
        print(f"\n  ✓ No missing MRI cases — ready to train!")

    print("=" * 60)


if __name__ == "__main__":
    main()
