"""
data/load_mri.py

Loads T2W, ADC, HBV volumes from PI-CAI dataset.
Handles the fold-based structure:
  images/
    picai_public_images_fold0/
      10000/
        10000_1000000_t2w.mha
        10000_1000000_adc.mha
        10000_1000000_hbv.mha
    picai_public_images_fold1/
      ...

Each case returns a normalised 3-channel numpy array: [3, Z, H, W]
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# YOUR PATHS — these match your local setup
# ─────────────────────────────────────────────────────────────
IMAGE_DIR  = Path(r"F:\MOD002691 - FP\pi_cai_project\images")
MODALITIES = ["t2w", "adc", "hbv"]   # channel 0=T2W, 1=ADC, 2=HBV

# All 5 fold folder names
FOLD_NAMES = [
    "picai_public_images_fold0",
    "picai_public_images_fold1",
    "picai_public_images_fold2",
    "picai_public_images_fold3",
    "picai_public_images_fold4",
]


def find_patient_folder(patient_id: str,
                        image_dir: Path = IMAGE_DIR) -> Path:
    """
    Search across all fold folders to find where a patient lives.
    Returns the full path to that patient folder.

    e.g. returns: .../images/picai_public_images_fold2/10000/
    """
    for fold in FOLD_NAMES:
        candidate = image_dir / fold / patient_id
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Patient {patient_id} not found in any fold under {image_dir}\n"
        f"Checked folds: {FOLD_NAMES}"
    )


def load_single_volume(filepath: Path) -> np.ndarray:
    """
    Load a single .mha MRI volume and return as numpy array.
    Shape returned: (Z, H, W)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    img = sitk.ReadImage(str(filepath))
    arr = sitk.GetArrayFromImage(img)   # SimpleITK returns (Z, Y, X)
    return arr.astype(np.float32)


def normalise_volume(arr: np.ndarray) -> np.ndarray:
    """
    Z-score normalisation per volume.
    Clips to [1st, 99th] percentile first to remove intensity outliers.
    """
    p1  = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    arr = np.clip(arr, p1, p99)

    mean = arr.mean()
    std  = arr.std() + 1e-8   # avoid divide-by-zero

    return (arr - mean) / std


def resize_volume(arr: np.ndarray,
                  target_shape: tuple = (20, 256, 256)) -> np.ndarray:
    """
    Resize volume to target (Z, H, W) using SimpleITK linear interpolation.
    """
    tz, th, tw = target_shape
    oz, oh, ow = arr.shape

    if (oz, oh, ow) == (tz, th, tw):
        return arr   # already correct size

    sitk_img         = sitk.GetImageFromArray(arr)
    original_size    = sitk_img.GetSize()    # (W, H, Z) in SimpleITK
    original_spacing = sitk_img.GetSpacing()

    new_size    = [tw, th, tz]
    new_spacing = [
        original_spacing[i] * original_size[i] / new_size[i]
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(sitk_img)
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def load_mri_case(patient_id: str,
                  study_id: str,
                  image_dir: Path   = IMAGE_DIR,
                  target_shape: tuple = (20, 256, 256)) -> np.ndarray:
    """
    Load and stack T2W, ADC, HBV for one patient/study.

    Args:
        patient_id   : e.g. "10000"
        study_id     : e.g. "1000000"
        image_dir    : root images folder (contains fold subfolders)
        target_shape : (Z, H, W) — all volumes resized to this

    Returns:
        numpy array shape (3, Z, H, W) — float32, normalised
    """
    patient_folder = find_patient_folder(patient_id, image_dir)
    prefix         = f"{patient_id}_{study_id}"
    channels       = []

    for modality in MODALITIES:
        filepath = patient_folder / f"{prefix}_{modality}.mha"
        vol      = load_single_volume(filepath)
        vol      = normalise_volume(vol)
        vol      = resize_volume(vol, target_shape)
        channels.append(vol)

    return np.stack(channels, axis=0)   # (3, Z, H, W)


def get_all_case_ids(image_dir: Path = IMAGE_DIR) -> list:
    """
    Scan ALL fold folders and return list of (patient_id, study_id) tuples.
    Uses T2W files to determine available studies per patient.

    Returns:
        List of ("10000", "1000000") tuples
    """
    case_ids = []

    for fold_name in FOLD_NAMES:
        fold_path = image_dir / fold_name

        if not fold_path.exists():
            print(f"WARNING: Fold folder not found: {fold_path}")
            continue

        for patient_folder in sorted(fold_path.iterdir()):
            if not patient_folder.is_dir():
                continue

            patient_id = patient_folder.name

            # Use T2W files to find study IDs
            t2w_files = sorted(patient_folder.glob("*_t2w.mha"))
            for t2w_file in t2w_files:
                # filename: 10000_1000000_t2w.mha
                parts    = t2w_file.stem.split("_")   # ["10000","1000000","t2w"]
                study_id = parts[1]
                case_ids.append((patient_id, study_id))

    return case_ids


# ─────────────────────────────────────────────────────────────
# Sanity check — run directly: python -m mri_baseline.data.load_mri
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("PI-CAI MRI Loader — Sanity Check")
    print("=" * 50)

    # Step 1: Find all cases
    print(f"\nScanning: {IMAGE_DIR}")
    print("This may take 10-20 seconds...\n")

    all_cases = get_all_case_ids()
    print(f"Total cases found: {len(all_cases)}")

    if len(all_cases) == 0:
        print("\nERROR: No cases found!")
        print("Check that IMAGE_DIR points to your images folder")
        print(f"Currently set to: {IMAGE_DIR}")
        exit(1)

    # Show fold distribution
    print("\nChecking fold distribution...")
    for fold in FOLD_NAMES:
        fold_path = IMAGE_DIR / fold
        if fold_path.exists():
            count = len(list(fold_path.iterdir()))
            print(f"  {fold}: {count} patients")
        else:
            print(f"  {fold}: NOT FOUND")

    # Step 2: Load one case
    pid, sid = all_cases[0]
    print(f"\nLoading first case: patient={pid}  study={sid}")

    try:
        vol = load_mri_case(pid, sid)

        print(f"\nResults:")
        print(f"  Output shape  : {vol.shape}")     # (3, 20, 256, 256)
        print(f"  dtype         : {vol.dtype}")
        print(f"  Overall range : [{vol.min():.3f}, {vol.max():.3f}]")
        print(f"\n  Per-channel stats:")
        for i, name in enumerate(MODALITIES):
            ch = vol[i]
            print(f"    {name}: mean={ch.mean():.3f}  std={ch.std():.3f}  "
                  f"min={ch.min():.3f}  max={ch.max():.3f}")

        # Quick shape assertion
        assert vol.shape[0] == 3,  "Expected 3 channels"
        assert vol.ndim     == 4,  "Expected 4D array (C, Z, H, W)"
        assert vol.dtype    == np.float32

        print("\nSANITY CHECK PASSED")

    except FileNotFoundError as e:
        print(f"\nERROR loading case: {e}")
        print("Check your image files exist and are named correctly")