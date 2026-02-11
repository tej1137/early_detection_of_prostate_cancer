import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
)

# ==========================================
# CONFIGURATION
# ==========================================
# Input paths
train_csv = r"F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\preprocessed_data\preprocessed_train.csv"
val_csv   = r"F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\preprocessed_data\preprocessed_val.csv"

# Single Fixed Output Directory
out_dir = r"F:\MOD002691 - FP\output_files\feature_combo"

# Create directory (if it doesn't exist)
os.makedirs(out_dir, exist_ok=True)
print(f"📂 Saving experiment results to: {out_dir}")

# Define output file paths
results_csv = os.path.join(out_dir, "feature_combo_results.csv")
results_xlsx = os.path.join(out_dir, "feature_combo_results.xlsx")

# ==========================================
# DATA LOADING
# ==========================================
target = "case_csPCa"

train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)

# Candidate features
candidate_features = ["psa_sc", "psad_sc", "patient_age_sc", "prostate_volume_sc"]
available_features = [c for c in candidate_features if c in train_df.columns]
print(f"Features available: {available_features}")

# ==========================================
# EXPERIMENT LOOP
# ==========================================
rows = []

for k in [1, 2, 3, 4]:
    for feats in itertools.combinations(available_features, k):
        feats = list(feats)
        combo_name = "+".join(feats)
        print(f"Testing: {combo_name}")

        X_train = train_df[feats]
        y_train = train_df[target].astype(int)

        X_val = val_df[feats]
        y_val = val_df[target].astype(int)

        # Train Logistic Regression Baseline
        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predictions
        y_proba = model.predict_proba(X_val)[:, 1]
        threshold = 0.5
        y_pred = (y_proba >= threshold).astype(int)

        # Metrics
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        roc = roc_auc_score(y_val, y_proba)
        ap  = average_precision_score(y_val, y_proba)

        rows.append({
            "features": combo_name,
            "k": k,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision_class1": precision_score(y_val, y_pred, pos_label=1, zero_division=0),
            "recall_class1": recall_score(y_val, y_pred, pos_label=1, zero_division=0),
            "f1_class1": f1_score(y_val, y_pred, pos_label=1, zero_division=0),
            "roc_auc": roc,
            "pr_auc_ap": ap,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })

        # ---- Plot Confusion Matrix ----
        plt.figure(figsize=(4.6, 4.0), dpi=140)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix (VAL)\n{combo_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"CM_{k}_{combo_name}.png"), dpi=200)
        plt.close()

        # ---- Plot ROC Curve ----
        plt.figure(figsize=(4.8, 4.0), dpi=140)
        RocCurveDisplay.from_predictions(y_val, y_proba, name="LogReg")
        plt.title(f"ROC (VAL) AUC={roc:.3f}\n{combo_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ROC_{k}_{combo_name}.png"), dpi=200)
        plt.close()

# ==========================================
# SAVE MASTER RESULTS
# ==========================================
results_df = pd.DataFrame(rows).sort_values(["roc_auc", "pr_auc_ap"], ascending=False)

# Save to CSV
results_df.to_csv(results_csv, index=False)
print(f"✅ CSV saved: {results_csv}")

# Save to Excel
results_df.to_excel(results_xlsx, index=False, sheet_name="Feature_Combinations")
print(f"✅ Excel saved: {results_xlsx}")

print("\nTop 5 Combinations (ROC-AUC):")
print(results_df.head(5)[["features", "roc_auc"]].to_string(index=False))
