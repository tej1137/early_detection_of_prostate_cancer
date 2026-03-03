"""
preprocessing/preprocess_clinical.py

ONE-TIME preprocessing script for PI-CAI clinical data.

What this does:
  Loads raw marksheet.csv → cleans → validates → saves clean CSV + norm stats

Run once:
  python -m mri_baseline.preprocessing.preprocess_clinical

Outputs:
  /workspace/data/preprocessed/clinical_preprocessed.csv
  /workspace/data/preprocessed/norm_stats.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# Edit paths here for local vs RunPod
# ══════════════════════════════════════════════════════════

class Config:
    # ── Input ──────────────────────────────────────────────
    marksheet_path = Path("F:/MOD002691 - FP/pi_cai_project/picai_labels/clinical_information/marksheet.csv")

    # ── Output ─────────────────────────────────────────────
    output_dir   = Path("F:/MOD002691 - FP/pi_cai_project/picai_labels/clinical_information/preprocessed")
    output_csv   = output_dir / "clinical_preprocessed.csv"
    output_stats = output_dir / "norm_stats.json"

    # ── Features ───────────────────────────────────────────
    # These 4 features feed into PSAEncoder
    clinical_features = ["psa", "psad", "prostate_volume", "patient_age"]
    target_col        = "case_csPCa"

    # The PSA trio — mathematically dependent:
    # PSAD = PSA / prostate_volume
    psa_trio = ["psa", "psad", "prostate_volume"]

    # ── Validation thresholds ──────────────────────────────
    # Max allowed error between reported PSAD and computed PSA/volume
    # Cases above this threshold have unreliable PSAD values
    psad_consistency_threshold = 1.0   # ng/mL²  (was 6.93 in raw data)

    # ── Known missing MRI cases ────────────────────────────
    # These patients have no MRI files on disk at all.
    # We exclude them here so the same clean CSV works for
    # both PSA-only and MRI+PSA training.
    known_missing_mri = ["10121_1000121", "10089_1000089"]
    # Note: 10403 has no study_id known — handled by MRI audit

    # ── Train/val/test split ratios ─────────────────────────
    train_ratio = 0.70
    val_ratio   = 0.15
    # test_ratio  = 0.15  (remainder)
    random_seed = 42

    def __init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# STEP 1 — LOAD RAW DATA
# ══════════════════════════════════════════════════════════

def load_raw(config: Config) -> pd.DataFrame:
    """
    Load marksheet.csv and create a case_id index.

    case_id format: "{patient_id}_{study_id}"
    e.g. patient 10000, study 1000000 → "10000_1000000"

    This is the unique identifier used across all scripts
    to link MRI files to clinical data.
    """
    print("\n" + "═"*60)
    print("STEP 1 — LOADING RAW DATA")
    print("═"*60)

    df = pd.read_csv(config.marksheet_path)
    print(f"  Raw rows loaded: {len(df)}")

    # Create case_id as the primary key
    df["case_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
    df = df.set_index("case_id")

    print(f"  Columns available: {list(df.columns)}")
    print(f"\n  Missing values per feature:")
    for col in config.clinical_features:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"    {col:<20}: {n:>4} missing ({100*n/len(df):.1f}%)")

    return df


# ══════════════════════════════════════════════════════════
# STEP 2 — FIX PSA TRIO DEPENDENCIES
# ══════════════════════════════════════════════════════════

def fix_psa_trio(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Handle the mathematical dependency between PSA, PSAD, and prostate_volume.

    The relationship is:  PSAD = PSA / prostate_volume

    This means if any 2 of the 3 values are known, we can compute the 3rd.
    We use this to recover cases that would otherwise be dropped.

    3 recovery cases:
      Case A: PSA + volume known, PSAD missing  → compute PSAD = PSA / volume
      Case B: PSAD + volume known, PSA missing  → compute PSA  = PSAD × volume
      Case C: PSA + PSAD known, volume missing  → compute vol  = PSA / PSAD

    After recovery, any case still missing any of the 3 is dropped —
    we cannot reliably impute with less than 2 of the 3 values.
    """
    print("\n" + "═"*60)
    print("STEP 2 — FIXING PSA TRIO DEPENDENCIES")
    print("═"*60)

    # Count completeness before
    trio_present = df[config.psa_trio].notna().sum(axis=1)
    print(f"\n  Before recovery:")
    print(f"    All 3 present : {(trio_present == 3).sum()}")
    print(f"    2 of 3 present: {(trio_present == 2).sum()}  ← recoverable")
    print(f"    1 of 3 present: {(trio_present == 1).sum()}  ← will be dropped")
    print(f"    0 of 3 present: {(trio_present == 0).sum()}  ← will be dropped")

    computed = {"psad": 0, "psa": 0, "prostate_volume": 0}

    # Case A: compute PSAD from PSA / volume
    mask = df["psad"].isna() & df["psa"].notna() & df["prostate_volume"].notna()
    if mask.sum() > 0:
        df.loc[mask, "psad"] = df.loc[mask, "psa"] / df.loc[mask, "prostate_volume"]
        computed["psad"] = mask.sum()
        print(f"\n  ✓ Computed PSAD for {computed['psad']} cases  (PSA ÷ volume)")

    # Case B: compute PSA from PSAD × volume
    mask = df["psa"].isna() & df["psad"].notna() & df["prostate_volume"].notna()
    if mask.sum() > 0:
        df.loc[mask, "psa"] = df.loc[mask, "psad"] * df.loc[mask, "prostate_volume"]
        computed["psa"] = mask.sum()
        print(f"  ✓ Computed PSA  for {computed['psa']} cases  (PSAD × volume)")

    # Case C: compute volume from PSA / PSAD (guard against divide by zero)
    mask = df["prostate_volume"].isna() & df["psa"].notna() & df["psad"].notna()
    if mask.sum() > 0:
        safe = mask & (df["psad"] > 0)
        df.loc[safe, "prostate_volume"] = df.loc[safe, "psa"] / df.loc[safe, "psad"]
        computed["prostate_volume"] = safe.sum()
        print(f"  ✓ Computed vol  for {computed['prostate_volume']} cases  (PSA ÷ PSAD)")

    total_computed = sum(computed.values())
    print(f"\n  Total values recovered: {total_computed}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 3 — DROP INCOMPLETE CASES
# ══════════════════════════════════════════════════════════

def drop_incomplete(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Drop any case that still has missing PSA trio values after recovery.

    We do NOT impute with mean/median here because the PSA trio is
    mathematically dependent — imputing one independently would break
    the PSA = PSAD × volume relationship and introduce false data.

    Age is independent so we impute that with the training median.
    """
    print("\n" + "═"*60)
    print("STEP 3 — DROPPING INCOMPLETE CASES")
    print("═"*60)

    before = len(df)

    # Drop cases where any PSA trio value is still missing
    trio_complete = df[config.psa_trio].notna().all(axis=1)
    df = df[trio_complete].copy()

    dropped = before - len(df)
    print(f"  Dropped {dropped} cases with incomplete PSA trio")
    print(f"  Remaining: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 4 — FIX PSAD INCONSISTENCIES
# ══════════════════════════════════════════════════════════

def fix_psad_consistency(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Check that reported PSAD matches computed PSA / prostate_volume.

    In the raw data, PSAD sometimes disagrees with PSA/volume by up to 6.93.
    This happens when:
      - Different units used (mL vs cc, ng/mL vs µg/L)
      - Data entry errors in the source database
      - Volume measured at a different time than PSA

    Strategy:
      - For cases where error < threshold: RECOMPUTE PSAD from PSA/volume
        (makes everything mathematically consistent)
      - For cases where error >= threshold: DROP the case
        (data is too unreliable to trust any of the values)
    """
    print("\n" + "═"*60)
    print("STEP 4 — FIXING PSAD CONSISTENCY")
    print("═"*60)

    # Compute what PSAD should be given PSA and volume
    computed_psad = df["psa"] / df["prostate_volume"]
    error = (df["psad"] - computed_psad).abs()

    consistent   = (error < config.psad_consistency_threshold).sum()
    inconsistent = (error >= config.psad_consistency_threshold).sum()

    print(f"\n  Consistency threshold: {config.psad_consistency_threshold} ng/mL²")
    print(f"  Consistent cases  (error < threshold): {consistent}")
    print(f"  Inconsistent cases (error ≥ threshold): {inconsistent}  ← will be dropped")
    print(f"  Max error in dataset: {error.max():.4f}")

    # For consistent cases: recompute PSAD to ensure mathematical consistency
    consistent_mask = error < config.psad_consistency_threshold
    df.loc[consistent_mask, "psad"] = computed_psad[consistent_mask]
    print(f"\n  ✓ Recomputed PSAD for {consistent} consistent cases")

    # Drop inconsistent cases
    before = len(df)
    df = df[consistent_mask].copy()
    print(f"  ✓ Dropped {before - len(df)} inconsistent cases")
    print(f"  Remaining: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 5 — HANDLE AGE
# ══════════════════════════════════════════════════════════

def handle_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age is the only feature we impute — it's independent of PSA/PSAD/volume
    so median imputation is safe here.

    We use the GLOBAL median (not train-only) for this preprocessing script.
    During training, normalisation stats are computed on train set only.
    """
    print("\n" + "═"*60)
    print("STEP 5 — HANDLING AGE")
    print("═"*60)

    missing = df["patient_age"].isna().sum()
    if missing > 0:
        median_age = df["patient_age"].median()
        df["patient_age"] = df["patient_age"].fillna(median_age)
        print(f"  Imputed {missing} missing ages with median: {median_age:.1f}")
    else:
        print(f"  No missing age values ✓")
    return df


# ══════════════════════════════════════════════════════════
# STEP 6 — REMOVE KNOWN MISSING MRI CASES
# ══════════════════════════════════════════════════════════

def remove_missing_mri(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Remove cases where MRI files are confirmed missing from disk.

    Even though this is a clinical preprocessing script, we remove
    these cases here so that ONE clean CSV works for all training scripts:
      - PSA-only training  → just ignores the MRI column, still benefits from clean data
      - MRI training       → won't encounter missing files
      - Fusion training    → same

    Any new missing cases found during MRI preprocessing will be added
    to config.known_missing_mri and this script re-run.
    """
    print("\n" + "═"*60)
    print("STEP 6 — REMOVING KNOWN MISSING MRI CASES")
    print("═"*60)

    before = len(df)
    to_remove = [cid for cid in config.known_missing_mri if cid in df.index]
    df = df.drop(index=to_remove)

    print(f"  Known missing MRI cases: {config.known_missing_mri}")
    print(f"  Actually found and removed: {to_remove}")
    print(f"  Rows before: {before}  →  After: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 7 — ENCODE TARGET
# ══════════════════════════════════════════════════════════

def encode_target(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("STEP 7 — ENCODING TARGET LABELS")
    print("═"*60)

    col = df[config.target_col]

    # Check for string dtype — covers both 'object' AND Arrow StringDtype
    is_string_dtype = pd.api.types.is_string_dtype(col) or pd.api.types.is_object_dtype(col)

    if is_string_dtype:
        # Normalise to uppercase to handle "yes"/"no"/"Yes" etc.
        df[config.target_col] = col.str.upper().map({"YES": 1, "NO": 0})
        print(f"  Mapped YES/NO → 1/0")

    # Final cast — use nullable Int64 first to catch any NaN from bad map values
    unmapped = df[config.target_col].isna().sum()
    if unmapped > 0:
        print(f"  ⚠️  WARNING: {unmapped} values could not be mapped — check raw column values:")
        print(f"    {col.unique()}")

    df[config.target_col] = df[config.target_col].astype(int)

    cancer = df[config.target_col].sum()
    benign = (df[config.target_col] == 0).sum()
    print(f"  Cancer (1): {cancer} ({100*cancer/len(df):.1f}%)")
    print(f"  Benign (0): {benign} ({100*benign/len(df):.1f}%)")
    return df


# ══════════════════════════════════════════════════════════
# STEP 8 — TRAIN/VAL/TEST SPLIT
# ══════════════════════════════════════════════════════════

def stratified_split(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Stratified split that preserves cancer prevalence in each subset.

    Stratification ensures:
      Train: ~28% cancer
      Val:   ~28% cancer
      Test:  ~28% cancer

    The split column is saved into the CSV so every script uses
    EXACTLY the same split — no risk of data leakage from different
    scripts creating different splits.

    split values: 'train', 'val', 'test'
    """
    print("\n" + "═"*60)
    print("STEP 8 — STRATIFIED TRAIN/VAL/TEST SPLIT")
    print("═"*60)

    from sklearn.model_selection import train_test_split

    all_ids    = df.index.tolist()
    all_labels = df[config.target_col].tolist()

    # First split: train vs temp (val + test)
    train_ids, temp_ids, _, temp_labels = train_test_split(
        all_ids, all_labels,
        train_size=config.train_ratio,
        stratify=all_labels,
        random_state=config.random_seed
    )

    # Second split: val vs test from temp
    relative_val = config.val_ratio / (1 - config.train_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=relative_val,
        stratify=temp_labels,
        random_state=config.random_seed
    )

    # Save split into the DataFrame
    df["split"] = "train"
    df.loc[val_ids,  "split"] = "val"
    df.loc[test_ids, "split"] = "test"

    print(f"  Train: {len(train_ids)} cases  ({100*df.loc[train_ids, config.target_col].mean():.1f}% cancer)")
    print(f"  Val:   {len(val_ids)} cases  ({100*df.loc[val_ids,   config.target_col].mean():.1f}% cancer)")
    print(f"  Test:  {len(test_ids)} cases  ({100*df.loc[test_ids,  config.target_col].mean():.1f}% cancer)")

    return df


# ══════════════════════════════════════════════════════════
# STEP 9 — COMPUTE NORMALISATION STATS
# ══════════════════════════════════════════════════════════

def compute_norm_stats(df: pd.DataFrame, config: Config) -> dict:
    """
    Compute mean and std for each clinical feature using TRAIN SET ONLY.

    CRITICAL: We use only training cases to compute stats.
    Using val/test data to compute normalisation would be data leakage —
    the model would indirectly see information about val/test distributions.

    Saved to norm_stats.json so every script uses identical normalisation.
    These stats are also saved inside model checkpoints for inference.

    Format:
    {
        "psa":              {"mean": 8.43, "std": 6.21},
        "psad":             {"mean": 0.14, "std": 0.11},
        "prostate_volume":  {"mean": 58.2, "std": 29.4},
        "patient_age":      {"mean": 66.1, "std": 7.8}
    }
    """
    print("\n" + "═"*60)
    print("STEP 9 — COMPUTING NORMALISATION STATS (train set only)")
    print("═"*60)

    train_df = df[df["split"] == "train"]
    stats = {}

    print(f"\n  {'Feature':<20} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*38}")
    for col in config.clinical_features:
        mean = float(train_df[col].mean())
        std  = float(train_df[col].std() + 1e-8)  # epsilon prevents divide-by-zero
        stats[col] = {"mean": mean, "std": std}
        print(f"  {col:<20} {mean:>8.3f} {std:>8.3f}")

    return stats


# ══════════════════════════════════════════════════════════
# STEP 10 — VALIDATE & SAVE
# ══════════════════════════════════════════════════════════

def validate_and_save(df: pd.DataFrame, stats: dict, config: Config):
    """
    Final validation checks before saving:
      1. No missing values in any clinical feature
      2. All target labels are 0 or 1
      3. Split column present and correct
      4. PSAD mathematically consistent (max error < threshold)

    Saves:
      clinical_preprocessed.csv  — clean data with split column
      norm_stats.json            — normalisation stats for training scripts
    """
    print("\n" + "═"*60)
    print("STEP 10 — VALIDATION & SAVING")
    print("═"*60)

    errors = []

    # Check 1: No missing values
    missing = df[config.clinical_features].isna().sum().sum()
    if missing > 0:
        errors.append(f"❌ {missing} missing values remain in clinical features")
    else:
        print("  ✓ No missing values in clinical features")

    # Check 2: Target is binary
    unique_labels = df[config.target_col].unique()
    if not set(unique_labels).issubset({0, 1}):
        errors.append(f"❌ Non-binary labels found: {unique_labels}")
    else:
        print(f"  ✓ Target labels are binary (0/1)")

    # Check 3: Split column correct
    valid_splits = {"train", "val", "test"}
    if not set(df["split"].unique()).issubset(valid_splits):
        errors.append(f"❌ Invalid split values: {df['split'].unique()}")
    else:
        print(f"  ✓ Split column correct")

    # Check 4: PSAD consistency after recomputation
    computed = df["psa"] / df["prostate_volume"]
    max_err  = (df["psad"] - computed).abs().max()
    if max_err > config.psad_consistency_threshold:
        errors.append(f"❌ PSAD still inconsistent after fix (max error: {max_err:.4f})")
    else:
        print(f"  ✓ PSAD consistent (max error: {max_err:.6f})")

    # Raise if any errors
    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print(f"  {e}")
        raise ValueError("Preprocessing validation failed — see errors above")

    # Save CSV
    cols_to_save = config.clinical_features + [config.target_col, "split"]
    df[cols_to_save].to_csv(config.output_csv)
    print(f"\n  ✓ Saved: {config.output_csv}")
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {cols_to_save}")

    # Save norm stats
    with open(config.output_stats, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved: {config.output_stats}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("CLINICAL DATA PREPROCESSING")
    print("PI-CAI Marksheet → Clean CSV + Norm Stats")
    print("=" * 60)

    config = Config()

    # Run all steps in order
    df = load_raw(config)
    df = fix_psa_trio(df, config)
    df = drop_incomplete(df, config)
    df = fix_psad_consistency(df, config)
    df = handle_age(df)
    df = remove_missing_mri(df, config)
    df = encode_target(df, config)
    df = stratified_split(df, config)
    stats = compute_norm_stats(df, config)
    validate_and_save(df, stats, config)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  {config.output_csv}")
    print(f"  {config.output_stats}")
    print(f"\nNext step: run preprocess_mri.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
