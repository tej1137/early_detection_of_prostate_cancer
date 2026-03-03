"""
data/load_psa.py - CORRECTED VERSION

Loads and preprocesses clinical features from PI-CAI marksheet.csv

CRITICAL FIX: Proper handling of dependent features (PSA, PSAD, prostate_volume)
Since PSAD = PSA / prostate_volume, these are mathematically related.

Preprocessing logic:
1. If all 3 (PSA, PSAD, volume) present → keep
2. If 2 of 3 present → compute the missing one (3-pass loop)
3. If 1 or 0 present → DROP ROW (cannot compute reliably)
4. Age: independent, impute with median if missing

Features: PSA, PSAD, prostate_volume, patient_age
Target:   case_csPCa (0 = benign, 1 = clinically significant cancer)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Auto-detect environment (RunPod vs Windows) ──────────────
if Path("/workspace").exists():
    MARKSHEET_PATH = Path("/workspace/data/picai_labels/clinical_information/marksheet.csv")
else:
    MARKSHEET_PATH = Path(r"F:\\MOD002691 - FP\\pi_cai_project\\picai_labels\\clinical_information\\marksheet.csv")

CLINICAL_FEATURES = ["psa", "psad", "prostate_volume", "patient_age"]
TARGET_COL = "case_csPCa"
PSA_TRIO = ["psa", "psad", "prostate_volume"]


# ── NEW: Robust 3-pass PSA trio calculator ────────────────────
def calculate_psad_psa_prostate_volume(df, psa_col='psa', psad_col='psad', vol_col='prostate_volume'):
    """
    3-pass calculation handles chain dependencies:
    Pass 1 may compute PSAD → Pass 2 can then use it to compute PSA, etc.
    Guards against division by zero on vol and psad.
    """
    df = df.copy()
    for c in [psa_col, psad_col, vol_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    for _ in range(3):
        # PSAD = PSA / volume
        m = df[psad_col].isna() & df[psa_col].notna() & df[vol_col].notna() & (df[vol_col] > 0)
        df.loc[m, psad_col] = df.loc[m, psa_col] / df.loc[m, vol_col]

        # PSA = PSAD * volume
        m = df[psa_col].isna() & df[psad_col].notna() & df[vol_col].notna() & (df[vol_col] > 0)
        df.loc[m, psa_col] = df.loc[m, psad_col] * df.loc[m, vol_col]

        # volume = PSA / PSAD
        m = df[vol_col].isna() & df[psa_col].notna() & df[psad_col].notna() & (df[psad_col] > 0)
        df.loc[m, vol_col] = df.loc[m, psa_col] / df.loc[m, psad_col]

    return df


def load_clinical_data(marksheet_path: Path = MARKSHEET_PATH) -> pd.DataFrame:
    """
    Load and clean the PI-CAI marksheet with proper handling of dependencies.

    Returns:
        Cleaned DataFrame indexed by case_id
    """
    df = pd.read_csv(marksheet_path)

    # Create case_id
    df["case_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
    df = df.set_index("case_id")

    print(f"Raw data: {len(df)} cases")
    print(f"\nMissing values per feature:")
    for col in CLINICAL_FEATURES:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing} ({100*missing/len(df):.1f}%)")

    # ═══════════════════════════════════════════════════════
    # STEP 1: Handle PSA/PSAD/Volume Dependencies (3-pass)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 1: Computing missing values in PSA trio")
    print(f"{'='*60}")

    trio_present = df[PSA_TRIO].notna().sum(axis=1)
    print(f"\nDistribution of PSA trio completeness:")
    print(f"  All 3 present:  {(trio_present == 3).sum()} cases")
    print(f"  2 of 3 present: {(trio_present == 2).sum()} cases")
    print(f"  1 of 3 present: {(trio_present == 1).sum()} cases")
    print(f"  0 of 3 present: {(trio_present == 0).sum()} cases")

    before_missing = df[PSA_TRIO].isna().sum().sum()
    df = calculate_psad_psa_prostate_volume(df)  # ← robust 3-pass replacement
    after_missing = df[PSA_TRIO].isna().sum().sum()
    print(f"\n  ✓ Recovered {before_missing - after_missing} values via 3-pass PSA trio calculation")

    # ═══════════════════════════════════════════════════════
    # STEP 2: Drop Rows with Incomplete PSA Trio
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 2: Dropping rows with incomplete PSA trio")
    print(f"{'='*60}")

    trio_present_after = df[PSA_TRIO].notna().sum(axis=1)
    rows_before = len(df)
    df = df[trio_present_after == 3].copy()
    rows_after = len(df)
    dropped = rows_before - rows_after

    print(f"  Dropped {dropped} cases with incomplete PSA trio")
    print(f"  Remaining: {rows_after} cases")

    # ═══════════════════════════════════════════════════════
    # STEP 3: Handle Age (Independent Feature)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 3: Handling age (independent feature)")
    print(f"{'='*60}")

    age_missing = df["patient_age"].isna().sum()
    if age_missing > 0:
        age_median = df["patient_age"].median()
        print(f"  Age missing: {age_missing} cases")
        print(f"  Imputing with median: {age_median:.1f}")
        df["patient_age"] = df["patient_age"].fillna(age_median)
    else:
        print(f"  No missing age values")

    # ═══════════════════════════════════════════════════════
    # STEP 4: Encode Target Labels
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 4: Encoding target labels")
    print(f"{'='*60}")

    col_values = df[TARGET_COL].tolist()
    mapped = []
    for v in col_values:
        if str(v).strip().upper() == 'YES':
            mapped.append(1)
        elif str(v).strip().upper() == 'NO':
            mapped.append(0)
        else:
            try:
                mapped.append(int(float(str(v))))
            except:
                mapped.append(0)

    df[TARGET_COL] = mapped
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    cancer_count = df[TARGET_COL].sum()
    benign_count = (df[TARGET_COL] == 0).sum()
    print(f"  Cancer (1): {cancer_count} ({100*cancer_count/len(df):.1f}%)")
    print(f"  Benign (0): {benign_count} ({100*benign_count/len(df):.1f}%)")

    # ═══════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total cases: {len(df)}")
    print(f"  Features: {CLINICAL_FEATURES}")
    print(f"  Missing values: {df[CLINICAL_FEATURES].isna().sum().sum()} (should be 0)")

    assert df[CLINICAL_FEATURES].isna().sum().sum() == 0, "Missing values still present!"

    computed_psad_check = df["psa"] / df["prostate_volume"]
    psad_diff = (df["psad"] - computed_psad_check).abs()
    max_error = psad_diff.max()

    print(f"\n  PSAD consistency check:")
    print(f"    Max error: {max_error:.6f}")
    if max_error < 0.01:
        print(f"    ✓ PSAD values consistent with PSA/volume")
    else:
        print(f"    ⚠ Warning: Some PSAD values inconsistent (max error {max_error:.4f})")

    return df[CLINICAL_FEATURES + [TARGET_COL]]


def fit_normalisation(df: pd.DataFrame) -> dict:
    """
    Compute mean and std for each clinical feature.
    ONLY call this on TRAINING SET to prevent data leakage.
    """
    stats = {}
    for col in CLINICAL_FEATURES:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std":  float(df[col].std() + 1e-8)
        }
    return stats


def get_clinical_vector(case_id: str,
                        df: pd.DataFrame,
                        norm_stats: dict) -> np.ndarray:
    """Return a normalised feature vector for one case."""
    if case_id not in df.index:
        print(f"WARNING: {case_id} not in clinical data. Using zeros.")
        return np.zeros(len(CLINICAL_FEATURES), dtype=np.float32)

    row = df.loc[case_id]
    vec = []
    for col in CLINICAL_FEATURES:
        val = float(row[col])
        norm = (val - norm_stats[col]["mean"]) / norm_stats[col]["std"]
        vec.append(norm)

    return np.array(vec, dtype=np.float32)


def get_label(case_id: str, df: pd.DataFrame) -> int:
    """Return binary label (0 or 1) for a case."""
    if case_id not in df.index:
        raise KeyError(f"Case {case_id} not found in clinical data")
    return int(df.loc[case_id, TARGET_COL])


def split_case_ids(df: pd.DataFrame,
                   train_ratio: float = 0.70,
                   val_ratio:   float = 0.15,
                   seed:        int   = 42) -> tuple:
    """Stratified train / val / test split."""
    from sklearn.model_selection import train_test_split

    all_ids    = df.index.tolist()
    all_labels = df[TARGET_COL].tolist()

    train_ids, temp_ids, _, temp_labels = train_test_split(
        all_ids, all_labels,
        train_size=train_ratio,
        stratify=all_labels,
        random_state=seed
    )

    relative_val = val_ratio / (1 - train_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=relative_val,
        stratify=temp_labels,
        random_state=seed
    )

    print(f"\n{'='*60}")
    print("DATA SPLIT")
    print(f"{'='*60}")
    print(f"  Train : {len(train_ids)} cases")
    print(f"  Val   : {len(val_ids)} cases")
    print(f"  Test  : {len(test_ids)} cases")

    return train_ids, val_ids, test_ids


# ═════════════════════════════════════════════════════════
# Sanity Check
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_clinical_data()
    missing = df[CLINICAL_FEATURES].isna().sum().sum()
    assert missing == 0, "❌ Missing values found!"
    print(f"✓ No missing values")

    train_ids, val_ids, test_ids = split_case_ids(df)
    norm_stats = fit_normalisation(df.loc[train_ids])
    vec = get_clinical_vector(train_ids[0], df, norm_stats)
    print(f"✓ Sample vector: {vec}")
    print(f"✓ ALL CHECKS PASSED")
