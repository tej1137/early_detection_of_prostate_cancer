import os
import glob
import pandas as pd
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
import logging
import numpy as np

# ================= CONFIGURATION =================
IMAGES_ROOT = r"F:\MOD002691 - FP\pi_cai_project\images"
MASKS_ROOT = r"F:\MOD002691 - FP\pi_cai_project\picai_labels\csPCa_lesion_delineations\AI\Bosma22a\decompressed_files"
OUTPUT_CSV = r"F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\radiomics_features.csv"

# ================= SETUP =================
radiomics.logger.setLevel(logging.ERROR)

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllImageTypes()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('shape')

# --- AGGRESSIVE SETTINGS ---
extractor.settings['geometryTolerance'] = 100.0  # Massive tolerance to ignore origin diffs
extractor.settings['correctMask'] = True        # Resample mask if needed
extractor.settings['minimumROIDimensions'] = 1
extractor.settings['binWidth'] = 25

print("✅ Extractor initialized with AGGRESSIVE settings.")

# ================= FIND FILES (Same as before) =================
print("🔍 Scanning files...")
image_map = {}
mri_files = glob.glob(os.path.join(IMAGES_ROOT, "**", "*.mha"), recursive=True)
for path in mri_files:
    fname = os.path.basename(path)
    if "_t2w.mha" in fname:
        uid = fname.replace("_t2w.mha", "")
        if uid not in image_map: image_map[uid] = {}
        image_map[uid]['t2w'] = path
    elif "_adc.mha" in fname:
        uid = fname.replace("_adc.mha", "")
        if uid not in image_map: image_map[uid] = {}
        image_map[uid]['adc'] = path

mask_map = {}
for f in os.listdir(MASKS_ROOT):
    if f.endswith(".nii") or f.endswith(".nii.gz"):
        uid = f.replace(".nii.gz", "").replace(".nii", "")
        mask_map[uid] = os.path.join(MASKS_ROOT, f)

paired_cases = []
for uid, img_data in image_map.items():
    if 't2w' in img_data and uid in mask_map:
        paired_cases.append({
            'id': uid,
            't2w': img_data['t2w'],
            'adc': img_data.get('adc', None),
            'mask': mask_map[uid],
            'patient_id': uid.split('_')[0],
            'study_id': uid.split('_')[1] if '_' in uid else ""
        })

print(f"✅ Found {len(paired_cases)} cases.")

# ================= ROBUST PROCESSING =================
results = []

for i, case in enumerate(paired_cases):
    print(f"[{i+1}/{len(paired_cases)}] {case['id']}: ", end="", flush=True)
    
    try:
        # 1. Load Mask Manually First
        mask_path = case['mask']
        mask_img = sitk.ReadImage(mask_path)
        mask_arr = sitk.GetArrayFromImage(mask_img)
        
        # Check for content
        if np.sum(mask_arr) == 0:
            print("⏩ Skipped (Empty)")
            continue

        # 2. Manual Alignment Check (Optional but safer)
        # We pass the file paths to PyRadiomics, letting it use 'correctMask'
        # to handle the resampling since we enabled it above.
        
        row = {'patient_id': case['patient_id'], 'study_id': case['study_id']}
        
        # Extract T2W
        feats = extractor.execute(case['t2w'], mask_path)
        for k, v in feats.items():
            if not k.startswith("diagnostics"):
                row[f"t2w_{k}"] = float(v)
                
        # Extract ADC
        if case['adc']:
            # ADC often has different geometry than T2W. 
            # We must be careful. PyRadiomics might fail if ADC and Mask differ too much.
            try:
                feats_adc = extractor.execute(case['adc'], mask_path)
                for k, v in feats_adc.items():
                    if not k.startswith("diagnostics"):
                        row[f"adc_{k}"] = float(v)
            except Exception as e_adc:
                print(f"(ADC failed: {str(e_adc)[:30]}...) ", end="")

        results.append(row)
        print("✅ Done")
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")

# ================= SAVE =================
if results:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 Done! Features saved to:\n{OUTPUT_CSV}")
else:
    print("\n❌ No features extracted.")
