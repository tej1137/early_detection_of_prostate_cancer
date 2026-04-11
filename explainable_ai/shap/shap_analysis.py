"""
shap_analysis.py  —  Clinical Feature Attribution (SHAP + Gradient x Input)
CrossModal Fusion Model — Test Set

Saves 3 publication-quality figures:
    shap_outputs/
        fig1_beeswarm.png
        fig2_importance_bar.png
        fig3_cancer_vs_benign.png
        attribution_results.json

Run:
    cd "F:\\MOD002691 - FP"
    python explainable_ai/shap/shap_analysis.py
"""

import sys, json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed — saves directly to file
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

ROOT = Path("F:/MOD002691 - FP")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "early_detection_of_prostate_cancer"))

from mri_baseline.models.psa_encoder import PSAEncoder

try:
    import shap
    HAS_SHAP = True
    print("  ✓ SHAP library found")
except ImportError:
    HAS_SHAP = False
    print("  ⚠ SHAP not installed — using Gradient x Input instead")
    print("    pip install shap   to get official SHAP values")

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

CFG = {
    "model_ckpt"  : ROOT / "checkpoints" / "crossmodal" / "fusion_crossmodal_unfrozen.pt",
    "clinical_csv": ROOT / "pi_cai_project" / "picai_labels" / "clinical_information" / "preprocessed" / "clinical_preprocessed.csv",
    "norm_stats"  : ROOT / "pi_cai_project" / "picai_labels" / "clinical_information" / "preprocessed" / "norm_stats.json",
    "output_dir"  : ROOT / "shap_outputs",
    "mri_dim"     : 512,
    "clinical_dim": 128,
    "fusion_dim"  : 640,
}

FEATURE_NAMES = ["PSA", "PSAD", "Prostate Volume", "Age"]
FEATURE_COLS  = ["psa", "psad", "prostate_volume", "patient_age"]

CFG["output_dir"].mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# CLINICAL-ONLY MODEL  (no MRI encoder — avoids 14GB RAM)
# ══════════════════════════════════════════════════════════

class ClinicalBranch(torch.nn.Module):
    """
    Loads only the clinical encoder + fusion head + classifier
    from the CrossModal checkpoint.
    MRI branch replaced with fixed zero embedding.
    """
    def __init__(self):
        super().__init__()
        self.clinical_encoder = PSAEncoder(in_features=4,
                                           embedding_dim=CFG["clinical_dim"])
        self.fusion_head = torch.nn.Sequential(
            torch.nn.Linear(CFG["fusion_dim"], 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.0),       # eval mode → dropout off anyway
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.0),
        )
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, clinical):
        B         = clinical.shape[0]
        clin_feat = self.clinical_encoder(clinical)             # (B, 128)
        mri_feat  = torch.zeros(B, CFG["mri_dim"])              # (B, 512) fixed zeros
        fused     = torch.cat([mri_feat, clin_feat], dim=1)     # (B, 640)
        fused     = self.fusion_head(fused)                     # (B, 128)
        return self.classifier(fused)                           # (B, 2)

    def predict_proba(self, clinical):
        logits = self.forward(clinical)
        return torch.softmax(logits, dim=1)[:, 1]


def load_model():
    print("[1] Loading clinical branch from CrossModal checkpoint...")
    full_state = torch.load(CFG["model_ckpt"], map_location="cpu", weights_only=True)

    model = ClinicalBranch()

    # Map keys: strip mri_encoder keys, keep clinical_encoder + fusion_head + classifier
    new_state = {}
    for k, v in full_state.items():
        if k.startswith("mri_encoder."):
            continue                     # skip MRI encoder
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"  Missing keys   : {missing}")
    print(f"  Unexpected keys: {unexpected}")
    model.eval()
    print("  ✓ Clinical branch loaded\n")
    return model


# ══════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════

def load_data():
    print("[2] Loading test set clinical data...")
    df = pd.read_csv(CFG["clinical_csv"], index_col="case_id")
    with open(CFG["norm_stats"]) as f:
        stats = json.load(f)

    test_df = df[df["split"] == "test"].copy()

    feat_norm, feat_raw, labels = [], [], []
    for case_id in test_df.index:
        row  = test_df.loc[case_id]
        norm, raw = [], []
        for col in FEATURE_COLS:
            mean = stats[col]["mean"]
            std  = stats[col]["std"]
            norm.append((float(row[col]) - mean) / std)
            raw.append(float(row[col]))
        feat_norm.append(norm)
        feat_raw.append(raw)
        labels.append(int(row["case_csPCa"]))

    feat_norm = np.array(feat_norm, dtype=np.float32)
    feat_raw  = np.array(feat_raw,  dtype=np.float32)
    labels    = np.array(labels,    dtype=np.int32)

    print(f"  ✓ {len(labels)} test cases  "
          f"(Cancer: {labels.sum()}  Benign: {(labels==0).sum()})\n")
    return feat_norm, feat_raw, labels


# ══════════════════════════════════════════════════════════
# ATTRIBUTION
# ══════════════════════════════════════════════════════════

def compute_shap_values(model, feat_norm):
    """Use SHAP KernelExplainer on clinical-only branch."""
    print("[3a] Running SHAP KernelExplainer (100 test cases)...")

    def predict_fn(x):
        t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model.predict_proba(t).numpy()

    background  = shap.sample(feat_norm, 50, random_state=42)
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_vals   = explainer.shap_values(feat_norm[:100], nsamples=200)
    print(f"  ✓ SHAP values computed  shape={np.array(shap_vals).shape}\n")
    return np.array(shap_vals), feat_norm[:100]


def compute_gradient_attribution(model, feat_norm):
    """Gradient x Input — fallback when SHAP not installed."""
    print("[3b] Computing Gradient x Input attribution...")
    model.eval()
    all_attr, all_probs = [], []

    for i in range(0, len(feat_norm), 32):
        batch = torch.tensor(feat_norm[i:i+32]).requires_grad_(True)
        prob  = model.predict_proba(batch)
        prob.sum().backward()
        grad  = batch.grad.detach().numpy()
        inp   = batch.detach().numpy()
        all_attr.append(grad * inp)
        all_probs.append(prob.detach().numpy())

    attr  = np.vstack(all_attr)
    probs = np.concatenate(all_probs)
    print(f"  ✓ Attributions computed  shape={attr.shape}\n")
    return attr, probs


# ══════════════════════════════════════════════════════════
# FIGURE 1 — BEESWARM
# ══════════════════════════════════════════════════════════

def plot_beeswarm(attr_vals, feat_raw_subset, labels_subset, method):
    print("[4] Saving Figure 1 — beeswarm plot...")
    n_feat = len(FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(10, 6))

    for fi in range(n_feat):
        vals   = attr_vals[:, fi]
        fv     = feat_raw_subset[:, fi]
        fmin, fmax = fv.min(), fv.max()
        fn     = (fv - fmin) / (fmax - fmin + 1e-8)

        y_pos  = n_feat - 1 - fi
        np.random.seed(fi)
        jitter = np.random.uniform(-0.28, 0.28, len(vals))

        colors = cm.RdBu_r(fn.astype(float))
        ax.scatter(vals, y_pos + jitter,
                   c=colors, s=18, alpha=0.72, linewidths=0)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(FEATURE_NAMES[::-1], fontsize=13)
    ax.axvline(0, color="#888780", linewidth=0.9, linestyle="--")
    ax.set_xlabel(f"{method} value  (negative → benign  ·  positive → cancer)",
                  fontsize=11)
    ax.set_title(f"Clinical feature attribution — CrossModal fusion model\n"
                 f"Each dot = one test patient  ({method})", fontsize=12)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.spines[["top","right"]].set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Feature value  (low → high)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    plt.tight_layout()
    out = CFG["output_dir"] / "fig1_beeswarm.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out}\n")


# ══════════════════════════════════════════════════════════
# FIGURE 2 — IMPORTANCE BAR
# ══════════════════════════════════════════════════════════

def plot_importance_bar(attr_vals, method):
    print("[5] Saving Figure 2 — importance bar chart...")
    mean_abs = np.abs(attr_vals).mean(axis=0)
    order    = np.argsort(mean_abs)

    colors = ["#85B7EB"] * len(FEATURE_NAMES)
    colors[order[-1]] = "#185FA5"       # darkest for top feature

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        [FEATURE_NAMES[i] for i in order],
        mean_abs[order],
        color=[colors[i] for i in order],
        height=0.55,
    )

    for bar, val in zip(bars, mean_abs[order]):
        ax.text(val + 0.0004,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.set_xlabel(f"Mean |{method} value|", fontsize=11)
    ax.set_title(f"Clinical feature importance\n"
                 f"CrossModal fusion model  ·  test set  ({method})",
                 fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.set_xlim(0, mean_abs.max() * 1.25)

    plt.tight_layout()
    out = CFG["output_dir"] / "fig2_importance_bar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out}\n")


# ══════════════════════════════════════════════════════════
# FIGURE 3 — CANCER VS BENIGN
# ══════════════════════════════════════════════════════════

def plot_cancer_vs_benign(attr_vals, labels_subset, method):
    print("[6] Saving Figure 3 — cancer vs benign...")
    cidx = labels_subset == 1
    bidx = labels_subset == 0
    c_mean = attr_vals[cidx].mean(axis=0)
    b_mean = attr_vals[bidx].mean(axis=0)

    x     = np.arange(len(FEATURE_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, c_mean, width,
                   label=f"Cancer cases (n={cidx.sum()})",
                   color="#D85A30", alpha=0.88)
    bars2 = ax.bar(x + width/2, b_mean, width,
                   label=f"Benign cases (n={bidx.sum()})",
                   color="#378ADD", alpha=0.88)

    for bar in list(bars1) + list(bars2):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (0.0004 if v >= 0 else -0.0004),
                f"{v:+.4f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8)

    ax.axhline(0, color="#888780", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, fontsize=12)
    ax.set_ylabel(f"Mean {method} attribution", fontsize=11)
    ax.set_title(f"Clinical feature attribution — cancer vs benign\n"
                 f"Positive = pushes prediction toward cancer  ({method})",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)

    plt.tight_layout()
    out = CFG["output_dir"] / "fig3_cancer_vs_benign.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out}\n")

    return c_mean, b_mean


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Clinical Feature Attribution — CrossModal Fusion Model")
    print("=" * 60 + "\n")

    model             = load_model()
    feat_norm, feat_raw, labels = load_data()

    # ── Choose attribution method ──────────────────────────
    if HAS_SHAP:
        attr_vals, feat_subset = compute_shap_values(model, feat_norm)
        labels_subset = labels[:100]
        feat_raw_sub  = feat_raw[:100]
        method        = "SHAP"
    else:
        attr_vals, _ = compute_gradient_attribution(model, feat_norm)
        feat_subset  = feat_norm
        feat_raw_sub = feat_raw
        labels_subset = labels
        method       = "Grad×Input"

    # ── Importance summary ─────────────────────────────────
    mean_abs = np.abs(attr_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]

    print("[3c] Feature importance ranking:")
    print("-" * 50)
    for rank, fi in enumerate(order):
        cm_  = attr_vals[labels_subset == 1, fi].mean()
        bm_  = attr_vals[labels_subset == 0, fi].mean()
        dirn = "→ cancer" if cm_ > 0 else "→ benign"
        print(f"  {rank+1}. {FEATURE_NAMES[fi]:<18} "
              f"mean|attr|={mean_abs[fi]:.4f}  "
              f"cancer={cm_:+.4f}  {dirn}")
    print()

    # ── Save figures ───────────────────────────────────────
    plot_beeswarm(attr_vals, feat_raw_sub, labels_subset, method)
    plot_importance_bar(attr_vals, method)
    c_mean, b_mean = plot_cancer_vs_benign(attr_vals, labels_subset, method)

    # ── Save JSON ──────────────────────────────────────────
    results = {
        "method"          : method,
        "model"           : "CrossModal Fusion unfrozen (AUROC 0.8138)",
        "n_cases"         : int(len(attr_vals)),
        "feature_names"   : FEATURE_NAMES,
        "importance_ranking": [FEATURE_NAMES[i] for i in order],
        "mean_abs"        : {FEATURE_NAMES[i]: round(float(mean_abs[i]),6) for i in range(4)},
        "cancer_mean"     : {FEATURE_NAMES[i]: round(float(c_mean[i]),6) for i in range(4)},
        "benign_mean"     : {FEATURE_NAMES[i]: round(float(b_mean[i]),6) for i in range(4)},
    }
    with open(CFG["output_dir"] / "attribution_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"\n  Figures saved to: {CFG['output_dir']}")
    print("    fig1_beeswarm.png")
    print("    fig2_importance_bar.png")
    print("    fig3_cancer_vs_benign.png")
    print("    attribution_results.json")
    print()
    print("  Top feature:", FEATURE_NAMES[order[0]],
          f"(mean|attr|={mean_abs[order[0]]:.4f})")


if __name__ == "__main__":
    main()