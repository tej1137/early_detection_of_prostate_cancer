"""
gradcam_crossmodal.py

Grad-CAM Explainability for CrossModal Fusion Model.

What this script does:
    1. Loads CrossModal fusion model (best model, AUROC 0.8138)
    2. Loads test set cases from clinical CSV
    3. Runs inference on ALL test cases
    4. Selects 5 TRUE POSITIVE cases (cancer, predicted cancer, high confidence)
    5. For each case:
        a. Generates 3D Grad-CAM heatmap on MRI encoder
        b. Saves heatmap as .nii.gz for ITK-SNAP overlay
        c. Loads human expert lesion annotation
        d. Computes centroid distance (mm) between heatmap and annotation
        e. Saves side-by-side visualisation (T2W + Grad-CAM + annotation)
        f. Computes clinical feature attributions (Grad × Input)
    6. Saves summary report with all centroid distances

Output structure:
    gradcam_outputs/
    ├── case_10021_1000021/
    │   ├── t2w_gradcam_slices.png        ← T2W + heatmap overlay
    │   ├── annotation_comparison.png     ← T2W + heatmap + lesion mask
    │   ├── gradcam_heatmap.nii.gz        ← open in ITK-SNAP
    │   ├── clinical_attribution.png      ← PSA/PSAD/volume/age importance
    │   └── case_summary.json            ← centroid distance + prediction
    └── summary_report.json              ← all 5 cases + mean centroid distance

ITK-SNAP usage for each case:
    1. File → Open Main Image → ../images/.../case_t2w.mha
    2. Segmentation → Load → ../resampled/case.nii.gz  (green = actual lesion)
    3. Add Layer → gradcam_heatmap.nii.gz               (red = model attention)
    Compare visually — do they overlap?

Run:
    cd "F:\\MOD002691 - FP"
    python gradcam_crossmodal.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from scipy import ndimage

# ── Add project root to path ───────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mri_baseline.models.mri_encoder import MRIEncoder
from mri_baseline.models.psa_encoder import PSAEncoder


# ══════════════════════════════════════════════════════════
# CONFIGURATION — update paths if needed
# ══════════════════════════════════════════════════════════

CFG = {
    # ── Model ──────────────────────────────────────────────
    "model_ckpt"    : ROOT / "checkpoints" / "crossmodal" / "fusion_crossmodal_unfrozen.pt",

    # ── Clinical data ──────────────────────────────────────
    "clinical_csv"  : ROOT / "pi_cai_project" / "picai_labels" / "clinical_information" / "preprocessed" / "clinical_preprocessed.csv",
    "norm_stats"    : ROOT / "pi_cai_project" / "picai_labels" / "clinical_information" / "preprocessed" / "norm_stats.json",

    # ── MRI images ─────────────────────────────────────────
    "image_root"    : ROOT / "pi_cai_project" / "images",
    "mri_folds"     : [
        "picai_public_images_fold0",
        "picai_public_images_fold1",
        "picai_public_images_fold2",
        "picai_public_images_fold3",
        "picai_public_images_fold4",
    ],

    # ── Lesion annotations ─────────────────────────────────
    "annotation_dir": ROOT / "pi_cai_project" / "picai_labels" / "csPCa_lesion_delineations" / "human_expert" / "resampled",

    # ── Output ─────────────────────────────────────────────
    "output_dir"    : ROOT / "gradcam_outputs",

    # ── Model dims (must match training) ──────────────────
    "mri_dim"       : 512,
    "clinical_dim"  : 128,
    "fusion_dim"    : 640,    # 512 + 128

    # ── Preprocessing ──────────────────────────────────────
    "target_size"   : (20, 160, 160),
    "sequences"     : ["t2w", "adc", "hbv"],

    # ── Selection ──────────────────────────────────────────
    "n_cases"       : 5,      # number of true positive cases to explain
    "min_prob"      : 0.65,   # minimum cancer probability to select

    # ── Grad-CAM ───────────────────────────────────────────
    "target_layer_idx": 3,    # conv_blocks[3] — last conv block

    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}


# ══════════════════════════════════════════════════════════
# CROSSMODAL FUSION MODEL
# ══════════════════════════════════════════════════════════

class CrossModalFusionModel(torch.nn.Module):
    """
    Reconstructs the CrossModal fusion model for inference.
    Must match train_fusion_crossmodal.py architecture exactly.
    """

    def __init__(self):
        super().__init__()
        self.mri_encoder      = MRIEncoder(embedding_dim=CFG["mri_dim"])
        self.clinical_encoder = PSAEncoder(
            in_features   = 4,
            embedding_dim = CFG["clinical_dim"],
        )
        self.fusion_head = torch.nn.Sequential(
            torch.nn.Linear(CFG["fusion_dim"], 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.3),
        )
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, mri, clinical):
        mri_feat  = self.mri_encoder(mri)
        clin_feat = self.clinical_encoder(clinical)
        fused     = torch.cat([mri_feat, clin_feat], dim=1)
        fused     = self.fusion_head(fused)
        return self.classifier(fused)


# ══════════════════════════════════════════════════════════
# GRAD-CAM 3D
# ══════════════════════════════════════════════════════════

class GradCAM3D:
    """
    3D Grad-CAM for volumetric MRI encoder.

    Hooks into the last conv block of MRIEncoder.
    Generates a 3D heatmap same size as input MRI.
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self.fwd_handle  = target_layer.register_forward_hook(self._save_act)
        self.bwd_handle  = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, target_size):
        """Generate normalised 3D Grad-CAM heatmap."""
        # Global average pool gradients → weights per channel
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted sum of activation maps
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        # Upsample to input size
        cam = F.interpolate(
            cam, size=target_size,
            mode="trilinear", align_corners=False
        )
        cam = cam[0, 0]

        # Normalise to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()   # (D, H, W)

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


# ══════════════════════════════════════════════════════════
# MRI PREPROCESSING
# ══════════════════════════════════════════════════════════

def find_patient_folder(patient_id: str) -> Path:
    """Search all fold directories for patient folder."""
    for fold in CFG["mri_folds"]:
        candidate = CFG["image_root"] / fold / patient_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Patient folder not found for {patient_id} in any fold"
    )


def load_mha(path: Path):
    """Load .mha file → numpy array + SimpleITK image."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, img


def normalise_volume(arr: np.ndarray) -> np.ndarray:
    """Percentile clipping + min-max normalisation to [0,1]."""
    arr   = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = arr[arr > 0]
    if len(nonzero) == 0:
        return arr
    p1  = np.percentile(nonzero, 1)
    p99 = np.percentile(nonzero, 99)
    arr = np.clip(arr, p1, p99)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-8:
        arr = (arr - vmin) / (vmax - vmin)
    return arr.astype(np.float32)


def resize_volume(arr: np.ndarray, target: tuple) -> np.ndarray:
    """Resize 3D volume using trilinear interpolation."""
    t = torch.from_numpy(arr)[None, None].float()
    t = F.interpolate(t, size=target, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def preprocess_mri(case_id: str):
    """
    Load T2W + ADC + HBV → preprocess → return tensor + reference image.

    Returns:
        mri_tensor : (1, 3, D, H, W) float tensor
        t2w_sitk   : SimpleITK image (original T2W, for ITK-SNAP export)
        t2w_vol    : (D, H, W) numpy array (for visualisation)
    """
    patient_id = case_id.split("_")[0]
    folder     = find_patient_folder(patient_id)

    channels   = []
    t2w_sitk   = None

    for seq in CFG["sequences"]:
        path = folder / f"{case_id}_{seq}.mha"
        arr, sitk_img = load_mha(path)
        arr = normalise_volume(arr)
        arr = resize_volume(arr, CFG["target_size"])
        channels.append(arr)
        if seq == "t2w":
            t2w_sitk = sitk_img

    mri = np.stack(channels, axis=0)                  # (3, D, H, W)
    mri_tensor = torch.from_numpy(mri).float().unsqueeze(0)  # (1, 3, D, H, W)
    t2w_vol    = channels[0]                           # (D, H, W)

    return mri_tensor, t2w_sitk, t2w_vol


def preprocess_clinical(case_id: str, df: pd.DataFrame, stats: dict) -> torch.Tensor:
    """Normalise clinical features for one case."""
    features = ["psa", "psad", "prostate_volume", "patient_age"]
    row      = df.loc[case_id]
    vals     = []
    for col in features:
        mean = stats[col]["mean"]
        std  = stats[col]["std"]
        vals.append((float(row[col]) - mean) / std)
    return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)


# ══════════════════════════════════════════════════════════
# CENTROID DISTANCE
# ══════════════════════════════════════════════════════════

def load_annotation(case_id: str):
    """
    Load human expert lesion annotation for a case.

    Returns:
        mask_arr  : (D, H, W) binary numpy array
        mask_sitk : SimpleITK image
        found     : bool — whether annotation exists
    """
    ann_path = CFG["annotation_dir"] / f"{case_id}.nii.gz"
    if not ann_path.exists():
        print(f"  ⚠ No annotation found for {case_id}")
        return None, None, False

    mask_sitk = sitk.ReadImage(str(ann_path))
    mask_arr  = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)
    mask_arr  = (mask_arr > 0).astype(np.float32)   # binarise
    return mask_arr, mask_sitk, True


def compute_centroid_world(mask_arr: np.ndarray, sitk_img) -> np.ndarray:
    """
    Compute centroid of a binary mask in world coordinates (mm).

    Uses SimpleITK to convert voxel centroid → physical coordinates.
    This accounts for voxel spacing and image orientation.

    Returns:
        centroid_mm : (3,) array [x, y, z] in mm
    """
    # Voxel centroid
    labeled, n = ndimage.label(mask_arr)
    if n == 0:
        return None

    # Use largest connected component
    sizes = ndimage.sum(mask_arr, labeled, range(1, n+1))
    largest = np.argmax(sizes) + 1
    component = (labeled == largest).astype(np.float32)

    centroid_vox = ndimage.center_of_mass(component)  # (z, y, x) in numpy

    # Convert to world coordinates using SimpleITK
    # SimpleITK uses (x, y, z) ordering
    centroid_world = sitk_img.TransformContinuousIndexToPhysicalPoint([
        float(centroid_vox[2]),  # x
        float(centroid_vox[1]),  # y
        float(centroid_vox[0]),  # z
    ])
    return np.array(centroid_world)


def compute_gradcam_centroid_world(cam_vol: np.ndarray, t2w_sitk, threshold: float = 0.5) -> np.ndarray:
    """
    Compute centroid of high-activation Grad-CAM region in world coordinates.

    Strategy:
        1. Threshold heatmap → binary mask in preprocessed space (20,160,160)
        2. Find centroid in preprocessed voxel space
        3. Scale voxel coords proportionally to original T2W voxel space
        4. Use SimpleITK to convert original voxel → world (mm)

    The key fix: we use the CENTRE of the preprocessed volume as the crop
    reference, matching how preprocess_mri.py crops/pads the original image.
    """
    # Threshold heatmap — use adaptive threshold if 0.5 finds nothing
    binary = (cam_vol > threshold).astype(np.float32)
    if binary.sum() == 0:
        binary = (cam_vol > cam_vol.mean()).astype(np.float32)
    if binary.sum() == 0:
        peak = np.unravel_index(cam_vol.argmax(), cam_vol.shape)
        centroid_prep = np.array(peak, dtype=float)
    else:
        centroid_prep = np.array(
            ndimage.center_of_mass(binary), dtype=float
        )  # (z, y, x) in preprocessed space

    # Original T2W dimensions
    orig_size = t2w_sitk.GetSize()   # SimpleITK: (x, y, z)
    orig_W    = float(orig_size[0])  # x
    orig_H    = float(orig_size[1])  # y
    orig_D    = float(orig_size[2])  # z

    prep_D, prep_H, prep_W = [float(v) for v in CFG["target_size"]]

    # Scale centroid from preprocessed → original voxel space
    # preprocess_mri.py uses centre-crop/pad so origin aligns at centre
    # Preprocessed centroid 0 → original centre - prep/2
    # Formula: orig_vox = orig_centre + (prep_vox - prep_centre) * scale

    orig_centre_z = orig_D / 2.0
    orig_centre_y = orig_H / 2.0
    orig_centre_x = orig_W / 2.0

    prep_centre_z = prep_D / 2.0
    prep_centre_y = prep_H / 2.0
    prep_centre_x = prep_W / 2.0

    scale_z = orig_D / prep_D
    scale_y = orig_H / prep_H
    scale_x = orig_W / prep_W

    orig_vox_z = orig_centre_z + (centroid_prep[0] - prep_centre_z) * scale_z
    orig_vox_y = orig_centre_y + (centroid_prep[1] - prep_centre_y) * scale_y
    orig_vox_x = orig_centre_x + (centroid_prep[2] - prep_centre_x) * scale_x

    # Clamp to valid voxel range
    orig_vox_z = float(np.clip(orig_vox_z, 0, orig_D - 1))
    orig_vox_y = float(np.clip(orig_vox_y, 0, orig_H - 1))
    orig_vox_x = float(np.clip(orig_vox_x, 0, orig_W - 1))

    # Convert original voxel → world coordinates (mm)
    # SimpleITK uses (x, y, z) ordering
    centroid_world = t2w_sitk.TransformContinuousIndexToPhysicalPoint([
        orig_vox_x,
        orig_vox_y,
        orig_vox_z,
    ])
    return np.array(centroid_world)


def euclidean_distance_mm(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two world coordinate points in mm."""
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


# ══════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════

def overlay_heatmap(gray: np.ndarray, heat: np.ndarray, alpha: float = 0.45):
    """Overlay Grad-CAM heatmap on grayscale MRI slice."""
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    heat_rgb = plt.get_cmap("jet")(heat.astype(np.float32))[..., :3]
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    return np.clip((1 - alpha) * gray_rgb + alpha * heat_rgb, 0, 1)


def select_top_slices(cam_vol: np.ndarray, n: int = 3) -> list:
    """Select n slices with highest Grad-CAM activation (avoid edges)."""
    D  = cam_vol.shape[0]
    lo = max(0, int(D * 0.2))
    hi = min(D, int(D * 0.9))
    scores    = cam_vol.reshape(D, -1).max(axis=1)
    top_local = np.argsort(scores[lo:hi])[-n:][::-1]
    return (top_local + lo).tolist()


def save_gradcam_figure(
    case_id   : str,
    t2w_vol   : np.ndarray,
    cam_vol   : np.ndarray,
    mask_arr  : np.ndarray,
    top_idx   : list,
    prob      : float,
    dist_mm   : float,
    out_dir   : Path,
):
    """
    Save 3-panel figure per slice:
    Left:   T2W MRI
    Middle: T2W + Grad-CAM heatmap
    Right:  T2W + lesion annotation (if available)
    """
    n_slices = len(top_idx)
    has_mask = mask_arr is not None

    fig, axes = plt.subplots(
        n_slices, 3 if has_mask else 2,
        figsize=(15 if has_mask else 10, 5 * n_slices)
    )
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(top_idx):
        # Resize mask slice to match preprocessed MRI if needed
        if has_mask:
            mask_slice = resize_mask_slice(mask_arr, s, t2w_vol.shape)
        else:
            mask_slice = None

        # Panel 1 — T2W only
        axes[row, 0].imshow(t2w_vol[s], cmap="gray")
        axes[row, 0].set_title(f"T2W — slice {s}", fontsize=11)
        axes[row, 0].axis("off")

        # Panel 2 — T2W + Grad-CAM
        axes[row, 1].imshow(overlay_heatmap(t2w_vol[s], cam_vol[s]))
        axes[row, 1].set_title(f"Grad-CAM — slice {s}", fontsize=11)
        axes[row, 1].axis("off")

        # Panel 3 — T2W + lesion annotation
        if has_mask and mask_slice is not None:
            axes[row, 2].imshow(t2w_vol[s], cmap="gray")
            if mask_slice.max() > 0:
                axes[row, 2].imshow(
                    mask_slice, cmap="Greens", alpha=0.5,
                    vmin=0, vmax=1
                )
            axes[row, 2].set_title(f"Lesion annotation — slice {s}", fontsize=11)
            axes[row, 2].axis("off")

    dist_str = f"{dist_mm:.1f} mm" if dist_mm is not None else "N/A"
    fig.suptitle(
        f"Case: {case_id} | P(csPCa)={prob:.3f} | "
        f"Centroid distance: {dist_str}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "gradcam_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved gradcam_comparison.png")


def resize_mask_slice(mask_arr: np.ndarray, slice_idx: int, target_shape: tuple):
    """Resize annotation mask to match preprocessed MRI spatial dimensions."""
    D, H, W = target_shape
    # Resize full mask to target shape first
    mask_resized = resize_volume(mask_arr, (D, H, W))
    return (mask_resized[slice_idx] > 0.5).astype(np.float32)


def save_gradcam_nifti(cam_vol: np.ndarray, t2w_sitk, out_dir: Path):
    """
    Save Grad-CAM heatmap as .nii.gz in original T2W space.
    Open in ITK-SNAP as overlay on T2W for visual verification.
    """
    orig_size = t2w_sitk.GetSize()  # (x, y, z)
    orig_D    = orig_size[2]
    orig_H    = orig_size[1]
    orig_W    = orig_size[0]

    # Upsample heatmap to original T2W size
    cam_t = torch.from_numpy(cam_vol).float()[None, None]
    cam_t = F.interpolate(
        cam_t, size=(orig_D, orig_H, orig_W),
        mode="trilinear", align_corners=False
    )
    cam_np = cam_t[0, 0].cpu().numpy().astype(np.float32)

    # Create SimpleITK image with T2W metadata
    cam_sitk = sitk.GetImageFromArray(cam_np)
    cam_sitk.CopyInformation(t2w_sitk)

    out_path = out_dir / "gradcam_heatmap.nii.gz"
    sitk.WriteImage(cam_sitk, str(out_path))
    print(f"  ✓ Saved gradcam_heatmap.nii.gz → open in ITK-SNAP")
    return out_path


def save_clinical_attribution(clinical_grad, clinical_inp, raw_vals, out_dir: Path):
    """Bar chart of clinical feature attributions (Grad × Input)."""
    names  = ["PSA", "PSAD", "Volume", "Age"]
    attr   = (clinical_grad * clinical_inp).tolist()
    colors = ["crimson" if v > 0 else "steelblue" for v in attr]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, attr, color=colors)
    ax.axhline(0, color="black", linewidth=1)

    for i, (v, rv) in enumerate(zip(attr, raw_vals)):
        ax.text(
            i, v + (0.005 if v >= 0 else -0.005),
            f"{v:+.3f}\n(raw={rv:.2f})",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )

    cancer_patch = mpatches.Patch(color="crimson",   label="→ cancer")
    benign_patch = mpatches.Patch(color="steelblue", label="→ benign")
    ax.legend(handles=[cancer_patch, benign_patch])
    ax.set_title("Clinical Feature Attribution (Grad × Input)")
    ax.set_ylabel("Attribution score")
    fig.tight_layout()
    fig.savefig(out_dir / "clinical_attribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved clinical_attribution.png")

    return {n: float(v) for n, v in zip(names, attr)}


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Grad-CAM Explainability — CrossModal Fusion Model")
    print("=" * 60)
    print(f"  Device : {CFG['device']}")
    print(f"  Model  : {CFG['model_ckpt']}")
    print()

    CFG["output_dir"].mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────
    print("[1] Loading CrossModal fusion model...")
    model = CrossModalFusionModel()

    state = torch.load(CFG["model_ckpt"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # Disable inplace ReLU for Grad-CAM compatibility
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False

    model = model.to(CFG["device"])
    model.eval()
    print(f"  ✓ Model loaded")

    # ── Load clinical data ─────────────────────────────────
    print("\n[2] Loading clinical data...")
    df    = pd.read_csv(CFG["clinical_csv"], index_col="case_id")
    with open(CFG["norm_stats"], "r") as f:
        stats = json.load(f)

    test_df = df[df["split"] == "test"].copy()
    print(f"  ✓ {len(test_df)} test cases loaded")

    # ── Run inference on all test cases ───────────────────
    print("\n[3] Running inference on test set to find true positives...")

    results = []
    for case_id in test_df.index:
        label = int(test_df.loc[case_id, "case_csPCa"])
        if label == 0:
            continue   # only process cancer cases for TP selection

        try:
            mri, t2w_sitk, t2w_vol = preprocess_mri(case_id)
            clinical = preprocess_clinical(case_id, test_df, stats)

            mri      = mri.to(CFG["device"])
            clinical = clinical.to(CFG["device"])

            with torch.no_grad():
                logits = model(mri, clinical)
                probs  = torch.softmax(logits, dim=1)
                prob   = float(probs[0, 1].item())
                pred   = int(probs.argmax(dim=1).item())

            # True positive = cancer patient predicted as cancer
            if pred == 1 and prob >= CFG["min_prob"]:
                results.append({
                    "case_id": case_id,
                    "label"  : label,
                    "pred"   : pred,
                    "prob"   : prob,
                })
                print(f"  ✓ TP found: {case_id}  P(cancer)={prob:.3f}")

        except FileNotFoundError as e:
            print(f"  ⚠ Skipping {case_id}: {e}")
            continue

    # Sort by confidence — pick top N
    results.sort(key=lambda x: x["prob"], reverse=True)
    selected = results[:CFG["n_cases"]]

    print(f"\n  Selected {len(selected)} true positive cases")
    if len(selected) == 0:
        print("  ERROR: No true positives found. Check model checkpoint and data paths.")
        return

    # ── Hook Grad-CAM to last conv block ──────────────────
    target_layer = model.mri_encoder.conv_blocks[CFG["target_layer_idx"]].block[3]
    cam_helper   = GradCAM3D(model, target_layer)

    # ── Process each selected case ─────────────────────────
    print("\n[4] Generating Grad-CAM for selected cases...")
    all_summaries = []

    for case_info in selected:
        case_id = case_info["case_id"]
        prob    = case_info["prob"]

        print(f"\n  ── Case: {case_id}  P(cancer)={prob:.3f} ──")

        # Create output directory for this case
        case_dir = CFG["output_dir"] / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Load MRI
        mri, t2w_sitk, t2w_vol = preprocess_mri(case_id)
        clinical = preprocess_clinical(case_id, test_df, stats)

        mri      = mri.to(CFG["device"])
        clinical = clinical.to(CFG["device"])

        # Enable gradients
        mri.requires_grad_(True)
        clinical.requires_grad_(True)

        # Forward pass
        logits = model(mri, clinical)
        probs  = torch.softmax(logits, dim=1)

        # Backward pass for Grad-CAM (cancer class)
        model.zero_grad(set_to_none=True)
        logits[0, 1].backward(retain_graph=True)

        # Generate heatmap
        cam_vol = cam_helper.generate(target_size=CFG["target_size"])
        top_idx = select_top_slices(cam_vol, n=3)

        # Load annotation
        mask_arr, mask_sitk, has_ann = load_annotation(case_id)

        # Compute centroid distance
        dist_mm = None
        gradcam_centroid = None
        lesion_centroid  = None

        gradcam_centroid = compute_gradcam_centroid_world(cam_vol, t2w_sitk)

        if has_ann and mask_arr.sum() > 0:
            lesion_centroid = compute_centroid_world(mask_arr, mask_sitk)
            if lesion_centroid is not None and gradcam_centroid is not None:
                dist_mm = euclidean_distance_mm(gradcam_centroid, lesion_centroid)
                print(f"  Grad-CAM centroid : {gradcam_centroid.round(1)} mm")
                print(f"  Lesion centroid   : {lesion_centroid.round(1)} mm")
                print(f"  Distance          : {dist_mm:.1f} mm")
        else:
            print(f"  No annotation — centroid distance not computed")

        # Save visualisations
        save_gradcam_figure(
            case_id, t2w_vol, cam_vol, mask_arr,
            top_idx, prob, dist_mm, case_dir
        )

        # Save Grad-CAM as .nii.gz for ITK-SNAP
        save_gradcam_nifti(cam_vol, t2w_sitk, case_dir)

        # Clinical attribution
        clinical_grad = clinical.grad.detach().cpu().numpy()[0]
        clinical_inp  = clinical.detach().cpu().numpy()[0]
        raw_vals      = df.loc[case_id, ["psa", "psad", "prostate_volume", "patient_age"]].values.astype(float)

        attr_dict = save_clinical_attribution(
            clinical_grad, clinical_inp, raw_vals, case_dir
        )

        # Case summary
        summary = {
            "case_id"              : case_id,
            "true_label"           : "csPCa",
            "predicted"            : "csPCa",
            "p_cancer"             : round(prob, 4),
            "top_slices"           : top_idx,
            "gradcam_centroid_mm"  : gradcam_centroid.round(1).tolist() if gradcam_centroid is not None else None,
            "lesion_centroid_mm"   : lesion_centroid.round(1).tolist()  if lesion_centroid  is not None else None,
            "centroid_distance_mm" : round(dist_mm, 2) if dist_mm is not None else None,
            "has_annotation"       : has_ann,
            "clinical_attribution" : attr_dict,
            "itk_snap_files"       : {
                "main_image"  : f"../pi_cai_project/images/.../{ case_id}_t2w.mha",
                "gradcam"     : f"{case_id}/gradcam_heatmap.nii.gz",
                "annotation"  : f"../picai_labels/csPCa_lesion_delineations/human_expert/resampled/{case_id}.nii.gz",
            }
        }

        with open(case_dir / "case_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        all_summaries.append(summary)
        print(f"  ✓ Case {case_id} complete → {case_dir}")

    cam_helper.close()

    # ── Final summary report ───────────────────────────────
    print("\n[5] Saving summary report...")

    distances = [
        s["centroid_distance_mm"]
        for s in all_summaries
        if s["centroid_distance_mm"] is not None
    ]

    summary_report = {
        "model"                     : "CrossModal Fusion (unfrozen)",
        "test_auroc"                : 0.8138,
        "n_cases_explained"         : len(all_summaries),
        "cases"                     : all_summaries,
        "centroid_distance_mm"      : {
            "mean"   : round(float(np.mean(distances)), 2)  if distances else None,
            "std"    : round(float(np.std(distances)),  2)  if distances else None,
            "min"    : round(float(np.min(distances)),  2)  if distances else None,
            "max"    : round(float(np.max(distances)),  2)  if distances else None,
            "values" : [round(d, 2) for d in distances],
        },
        "interpretation" : {
            "< 10mm"  : "Excellent — model focused on correct region",
            "10-20mm" : "Good — model near the lesion",
            "20-30mm" : "Fair — some spatial error",
            "> 30mm"  : "Poor — model distracted by wrong region",
        }
    }

    with open(CFG["output_dir"] / "summary_report.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    # ── Print final results ────────────────────────────────
    print("\n" + "=" * 60)
    print("  GRAD-CAM RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Cases explained : {len(all_summaries)}")
    if distances:
        print(f"  Mean centroid distance : {np.mean(distances):.1f} ± {np.std(distances):.1f} mm")
        print(f"  Range                  : {np.min(distances):.1f} – {np.max(distances):.1f} mm")
        print()
        for s in all_summaries:
            d = s['centroid_distance_mm']
            d_str = f"{d:.1f} mm" if d else "N/A"
            print(f"  {s['case_id']}  P={s['p_cancer']:.3f}  dist={d_str}")

    print(f"\n  Output directory: {CFG['output_dir']}")
    print()
    print("  ITK-SNAP verification steps for each case:")
    print("  1. File → Open Main Image → [case]_t2w.mha")
    print("  2. Segmentation → Load → [case].nii.gz  (green = lesion)")
    print("  3. Tools → Image Layer Inspector → gradcam_heatmap.nii.gz")
    print("  4. Check if red/yellow heatmap overlaps green lesion")
    print("  5. Screenshot for dissertation Figure X")
    print("=" * 60)


if __name__ == "__main__":
    main()