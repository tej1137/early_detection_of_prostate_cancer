#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import SimpleITK as sitk

from mri_baseline.models.fusion_model import MultimodalFusionModel

# ============================================================
# 1) PATIENT CONFIG — updated to 10021
# ============================================================

CKPT_PATH = "checkpoints/best_fusion_model.pth"

T2W_PATH = r"F:\MOD002691 - FP\pi_cai_project\images\picai_public_images_fold4\10021\10021_1000021_t2w.mha"
ADC_PATH = r"F:\MOD002691 - FP\pi_cai_project\images\picai_public_images_fold4\10021\10021_1000021_adc.mha"
HBV_PATH = r"F:\MOD002691 - FP\pi_cai_project\images\picai_public_images_fold4\10021\10021_1000021_hbv.mha"

# Clinical values for 10021 from marksheet
PSA             = 23.0
PSAD            = 0.39
PROSTATE_VOLUME = 60.0
AGE             = 61.0

FALLBACK_MEAN = [7.5, 0.18, 45.0, 66.0]
FALLBACK_STD  = [4.2, 0.11, 18.0,  7.5]

OUT_DIR = "gradcam_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2) CONSTANTS
# ============================================================

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES   = ["Benign", "csPCa"]
POS_CLASS_IDX = 1
TARGET_SIZE   = (20, 256, 256)

# ============================================================
# 3) HELPERS
# ============================================================

def read_mha(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, img

def normalise_volume(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip(x, lo, hi)
    return (x - x.mean()) / (x.std() + 1e-8)

def resize_3d_np(arr, out_size=(20, 256, 256)):
    t = torch.from_numpy(arr)[None, None]
    t = F.interpolate(t, size=out_size, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy()

def preprocess_mri(t2w_path, adc_path, hbv_path):
    t2w_arr, t2w_img = read_mha(t2w_path)
    adc_arr, _       = read_mha(adc_path)
    hbv_arr, _       = read_mha(hbv_path)
    t2w = resize_3d_np(normalise_volume(t2w_arr), TARGET_SIZE)
    adc = resize_3d_np(normalise_volume(adc_arr), TARGET_SIZE)
    hbv = resize_3d_np(normalise_volume(hbv_arr), TARGET_SIZE)
    x = np.stack([t2w, adc, hbv], axis=0)
    x = torch.from_numpy(x).float().unsqueeze(0)
    return x, t2w_img

def clean_state_dict(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}

def extract_norm_stats(ckpt):
    for key in ["norm_stats", "normstats", "clinical_norm_stats",
                "feature_norm_stats", "tabular_norm_stats"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            d = ckpt[key]
            mean = d.get("mean", d.get("means"))
            std  = d.get("std",  d.get("stds"))
            if mean is not None and std is not None:
                return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)
    mean, std = None, None
    for k in ["clinical_mean", "feature_mean", "mean"]:
        if k in ckpt:
            mean = np.array(ckpt[k], dtype=np.float32)
            break
    for k in ["clinical_std", "feature_std", "std"]:
        if k in ckpt:
            std = np.array(ckpt[k], dtype=np.float32)
            break
    if mean is not None and std is not None and len(mean) == 4:
        return mean, std
    return (np.array(FALLBACK_MEAN, dtype=np.float32),
            np.array(FALLBACK_STD,  dtype=np.float32))

def preprocess_clinical(psa, psad, volume, age, mean, std):
    x = np.array([psa, psad, volume, age], dtype=np.float32)
    x = (x - mean) / (std + 1e-8)
    return torch.from_numpy(x).float().unsqueeze(0)

def disable_inplace_relu(module):
    for child in module.children():
        if isinstance(child, torch.nn.ReLU):
            child.inplace = False
        disable_inplace_relu(child)

# ============================================================
# 4) GRAD-CAM
# ============================================================

class GradCAM3D:
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
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

# ============================================================
# 5) VISUALISATION
# ============================================================

def overlay_slice(gray, heat, alpha=0.45):
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    heat_rgb = plt.get_cmap("jet")(heat.astype(np.float32))[..., :3]
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    return np.clip((1 - alpha) * gray_rgb + alpha * heat_rgb, 0, 1)

def select_top_slices(cam_vol, n=3):
    D  = cam_vol.shape[0]
    lo = max(0, int(D * 0.25))
    hi = min(D, int(D * 0.85))
    if hi <= lo:
        lo, hi = 0, D
    scores    = cam_vol.reshape(D, -1).max(axis=1)
    top_local = np.argsort(scores[lo:hi])[-n:][::-1]
    return (top_local + lo).tolist()

def save_top_slices(t2w_vol, cam_vol, pred_prob, top_idx):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for row, s in enumerate(top_idx):
        axes[row, 0].imshow(t2w_vol[s], cmap="gray")
        axes[row, 0].set_title(f"T2W slice {s}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(overlay_slice(t2w_vol[s], cam_vol[s]))
        axes[row, 1].set_title(f"Grad-CAM slice {s}")
        axes[row, 1].axis("off")
    fig.suptitle(
        f"Top MRI regions influencing csPCa decision | P(csPCa)={pred_prob:.3f}",
        fontsize=14
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gradcam_top_slices.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_gradcam_mha(cam_vol, reference_sitk_img):
    orig_size = reference_sitk_img.GetSize()
    orig_D, orig_H, orig_W = orig_size[2], orig_size[1], orig_size[0]
    cam_tensor = torch.from_numpy(cam_vol).float()[None, None]
    cam_resized = F.interpolate(
        cam_tensor,
        size=(orig_D, orig_H, orig_W),
        mode="trilinear",
        align_corners=False
    )
    cam_np = cam_resized[0, 0].cpu().numpy()
    cam_sitk = sitk.GetImageFromArray(cam_np.astype(np.float32))
    cam_sitk.CopyInformation(reference_sitk_img)
    out_path = os.path.join(OUT_DIR, "gradcam_overlay.mha")
    sitk.WriteImage(cam_sitk, out_path)
    print(f"  Saved ITK-SNAP overlay → {out_path}  (size: {orig_W}x{orig_H}x{orig_D})")

def save_clinical_attribution(attrib, raw_vals):
    names  = ["PSA", "PSAD", "Volume", "Age"]
    colors = ["crimson" if v > 0 else "steelblue" for v in attrib]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, attrib, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    for i, v in enumerate(attrib):
        ax.text(i, v + (0.01 if v >= 0 else -0.01),
                f"{v:.3f}\n(raw={raw_vals[i]:.2f})",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9)
    ax.set_title("Clinical feature influence on csPCa logit")
    ax.set_ylabel("Grad x Input")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "clinical_attribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 6) LOAD MODEL
# ============================================================

ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

state_dict = clean_state_dict(state_dict)
mean, std  = extract_norm_stats(ckpt if isinstance(ckpt, dict) else {})

model = MultimodalFusionModel()
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys   :", missing)
print("Unexpected keys:", unexpected)

disable_inplace_relu(model)
model = model.to(DEVICE)
model.eval()

target_layer = model.mri_encoder.conv_blocks[3].block[3]
cam_helper   = GradCAM3D(model, target_layer)

# ============================================================
# 7) PREP INPUTS
# ============================================================

mri, t2w_sitk_img = preprocess_mri(T2W_PATH, ADC_PATH, HBV_PATH)
mri               = mri.to(DEVICE)
clinical          = preprocess_clinical(PSA, PSAD, PROSTATE_VOLUME, AGE, mean, std).to(DEVICE)

mri.requires_grad_(True)
clinical.requires_grad_(True)

t2w_plot = mri[0, 0].detach().cpu().numpy()

# ============================================================
# 8) INFERENCE + EXPLANATION
# ============================================================

logits   = model(mri, clinical)
probs    = torch.softmax(logits, dim=1)
pred_idx = int(probs.argmax(dim=1).item())
pos_prob = float(probs[0, POS_CLASS_IDX].item())

print(f"\nPrediction : {CLASS_NAMES[pred_idx]}")
print(f"P(Benign)  = {float(probs[0,0]):.4f}")
print(f"P(csPCa)   = {float(probs[0,1]):.4f}")

model.zero_grad(set_to_none=True)
logits[0, POS_CLASS_IDX].backward(retain_graph=True)

cam_vol = cam_helper.generate(target_size=TARGET_SIZE)
top_idx = select_top_slices(cam_vol, n=3)

save_top_slices(t2w_plot, cam_vol, pos_prob, top_idx)
save_gradcam_mha(cam_vol, t2w_sitk_img)

clinical_grad = clinical.grad.detach().cpu().numpy()[0]
clinical_inp  = clinical.detach().cpu().numpy()[0]
clinical_attr = clinical_grad * clinical_inp
raw_vals      = np.array([PSA, PSAD, PROSTATE_VOLUME, AGE], dtype=np.float32)
save_clinical_attribution(clinical_attr, raw_vals)

feature_names = ["PSA", "PSAD", "Volume", "Age"]
order = np.argsort(-np.abs(clinical_attr))
print("\nTop clinical contributors:")
for i in order:
    direction = "→ cancer" if clinical_attr[i] > 0 else "→ benign"
    print(f"  {feature_names[i]:<8} attr={clinical_attr[i]:+.4f}  {direction}")

summary = {
    "prediction"    : CLASS_NAMES[pred_idx],
    "p_benign"      : float(probs[0, 0].item()),
    "p_cspca"       : float(probs[0, 1].item()),
    "top_slices"    : top_idx,
    "clinical_attr" : {n: float(clinical_attr[i]) for i, n in enumerate(feature_names)}
}

with open(os.path.join(OUT_DIR, "explanation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

cam_helper.close()

print(f"\n✓ Saved to: {OUT_DIR}/")
print("  gradcam_top_slices.png")
print("  gradcam_overlay.mha    ← open in ITK-SNAP with T2W for validation")
print("  clinical_attribution.png")
print("  explanation_summary.json")
