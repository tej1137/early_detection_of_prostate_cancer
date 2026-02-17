"""
data/load_psa.py

Loads and preprocesses clinical features from PI-CAI marksheet.csv
Features: PSA, PSAD, prostate_volume, patient_age
Target:   case_csPCa (0 = benign, 1 = clinically significant cancer)

Usage:
    from data.load_psa import load_clinical_data, get_clinical_vector
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG: Update to your local marksheet.csv path
# ─────────────────────────────────────────────
MARKSHEET_PATH = Path(r"F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\marksheet.csv")

CLINICAL_FEATURES = ["psa", "psad", "prostate_volume", "patient_age"]
TARGET_COL        = "case_csPCa"


def load_clinical_data(marksheet_path: Path = MARKSHEET_PATH) -> pd.DataFrame:
    """
    Load and clean the PI-CAI marksheet.

    Steps:
      1. Load CSV
      2. Create a unique case_id = patient_id + study_id
      3. Compute missing PSAD where PSA and volume are available
      4. Drop rows missing 2+ features (can't impute reliably)
      5. Impute remaining single-column NaNs with median
      6. Encode target as integer (0/1)

    Returns:
        Cleaned DataFrame indexed by case_id
    """
    df = pd.read_csv(marksheet_path)

    # ── create case_id ──────────────────────────────
    df["case_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
    df = df.set_index("case_id")

    # ── compute missing PSAD = PSA / prostate_volume ─
    missing_psad = df["psad"].isna() & df["psa"].notna() & df["prostate_volume"].notna()
    df.loc[missing_psad, "psad"] = (
        df.loc[missing_psad, "psa"] / df.loc[missing_psad, "prostate_volume"]
    )

    # ── drop rows missing 2+ of our core features ───
    missing_counts = df[CLINICAL_FEATURES].isna().sum(axis=1)
    df = df[missing_counts < 2].copy()

    # ── impute remaining NaNs with column median ─────
    for col in CLINICAL_FEATURES:
        median_val = df[col].median()
        df[col]    = df[col].fillna(median_val)

    # ── encode target ────────────────────────────────
    df[TARGET_COL] = df[TARGET_COL].map({"YES": 1, "NO": 0})
    # handle if already 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    print(f"Loaded {len(df)} cases after cleaning")
    print(f"  Cancer (1): {df[TARGET_COL].sum()}")
    print(f"  Benign (0): {(df[TARGET_COL] == 0).sum()}")

    return df[CLINICAL_FEATURES + [TARGET_COL]]


def fit_normalisation(df: pd.DataFrame) -> dict:
    """
    Compute mean and std for each clinical feature on a training set.
    Call this ONLY on your training split — not on val/test.

    Returns:
        dict of {feature: {"mean": float, "std": float}}
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
    """
    Return a normalised feature vector for one case.

    Args:
        case_id    : e.g. "10000_1000000"
        df         : cleaned DataFrame from load_clinical_data()
        norm_stats : stats dict from fit_normalisation()

    Returns:
        numpy array of shape (4,) — float32, z-score normalised
    """
    if case_id not in df.index:
        # Case has no clinical data — return zeros (model learns to ignore)
        print(f"WARNING: {case_id} not in clinical data. Using zeros.")
        return np.zeros(len(CLINICAL_FEATURES), dtype=np.float32)

    row = df.loc[case_id]
    vec = []
    for col in CLINICAL_FEATURES:
        val  = float(row[col])
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
                   seed: int = 42) -> tuple:
    """
    Stratified train / val / test split by label.
    Stratified = preserves cancer/benign ratio in each split.

    Returns:
        (train_ids, val_ids, test_ids) — lists of case_id strings
    """
    from sklearn.model_selection import train_test_split

    all_ids    = df.index.tolist()
    all_labels = df[TARGET_COL].tolist()

    # First split: train vs temp (val + test)
    train_ids, temp_ids, _, temp_labels = train_test_split(
        all_ids, all_labels,
        train_size=train_ratio,
        stratify=all_labels,
        random_state=seed
    )

    # Second split: val vs test from temp
    relative_val = val_ratio / (1 - train_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=relative_val,
        stratify=temp_labels,
        random_state=seed
    )

    print(f"\nSplit summary:")
    print(f"  Train : {len(train_ids)} cases")
    print(f"  Val   : {len(val_ids)} cases")
    print(f"  Test  : {len(test_ids)} cases")

    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────
# Quick sanity check — run this file directly
# python data/load_psa.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading clinical data...")
    df = load_clinical_data()

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nFeature statistics:")
    print(df[CLINICAL_FEATURES].describe().round(3))

    # Test normalisation
    train_ids, val_ids, test_ids = split_case_ids(df)

    train_df   = df.loc[train_ids]
    norm_stats = fit_normalisation(train_df)

    sample_id  = train_ids[0]
    vec        = get_clinical_vector(sample_id, df, norm_stats)
    label      = get_label(sample_id, df)

    print(f"\nSample case  : {sample_id}")
    print(f"Feature vec  : {vec}")
    print(f"Label        : {label}")
    print("\nSanity check PASSED")