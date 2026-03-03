import SimpleITK as sitk
import numpy as np
from pathlib import Path

IMAGE_DIR  = Path("/workspace/data/images")
MODALITIES = ["t2w", "adc", "hbv"]

FOLD_NAMES = [
    "picai_public_images_fold0",
    "picai_public_images_fold1",
    "picai_public_images_fold2",
    "picai_public_images_fold3",
    "picai_public_images_fold4",
]

def find_patient_folder(patient_id, image_dir=IMAGE_DIR):
    for fold in FOLD_NAMES:
        candidate = image_dir / fold / patient_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Patient {patient_id} not found in any fold under {image_dir}")

def load_single_volume(filepath):
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    img = sitk.ReadImage(str(filepath))
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(np.float32)

def normalise_volume(arr):
    p1  = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    arr = np.clip(arr, p1, p99)
    mean = arr.mean()
    std  = arr.std() + 1e-8
    return (arr - mean) / std

def resize_volume(arr, target_shape=(20, 256, 256)):
    tz, th, tw = target_shape
    oz, oh, ow = arr.shape
    if (oz, oh, ow) == (tz, th, tw):
        return arr
    sitk_img         = sitk.GetImageFromArray(arr)
    original_size    = sitk_img.GetSize()
    original_spacing = sitk_img.GetSpacing()
    new_size    = [tw, th, tz]
    new_spacing = [original_spacing[i] * original_size[i] / new_size[i] for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(sitk_img)
    return sitk.GetArrayFromImage(resampled).astype(np.float32)

def load_mri_case(patient_id, study_id, image_dir=IMAGE_DIR, target_shape=(20, 256, 256)):
    patient_folder = find_patient_folder(patient_id, image_dir)
    prefix         = f"{patient_id}_{study_id}"
    channels       = []
    for modality in MODALITIES:
        filepath = patient_folder / f"{prefix}_{modality}.mha"
        vol      = load_single_volume(filepath)
        vol      = normalise_volume(vol)
        vol      = resize_volume(vol, target_shape)
        channels.append(vol)
    return np.stack(channels, axis=0)

def get_all_case_ids(image_dir=IMAGE_DIR):
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
            t2w_files  = sorted(patient_folder.glob("*_t2w.mha"))
            for t2w_file in t2w_files:
                parts    = t2w_file.stem.split("_")
                study_id = parts[1]
                case_ids.append((patient_id, study_id))
    return case_ids

if __name__ == "__main__":
    print("Scanning:", IMAGE_DIR)
    all_cases = get_all_case_ids()
    print(f"Total cases found: {len(all_cases)}")
    if len(all_cases) == 0:
        print("ERROR: No cases found!")
        exit(1)
    for fold in FOLD_NAMES:
        fold_path = IMAGE_DIR / fold
        count = len(list(fold_path.iterdir())) if fold_path.exists() else 0
        print(f"  {fold}: {count} patients")
    pid, sid = all_cases[0]
    print(f"Loading first case: patient={pid} study={sid}")
    vol = load_mri_case(pid, sid)
    print(f"Shape: {vol.shape}, dtype: {vol.dtype}")
    assert vol.shape[0] == 3 and vol.ndim == 4
    print("SANITY CHECK PASSED")
