# save as: check_missing_files.py
# run with: python check_missing_files.py

import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────
MARKSHEET = Path("pi_cai_project/picai_labels/clinical_information/marksheet.csv")
IMAGE_ROOT = Path("pi_cai_project/images")
FOLDS = [f"picai_public_images_fold{i}" for i in range(5)]
SEQUENCES = ["t2w", "adc", "hbv"]  # adjust if your sequences are named differently
# ──────────────────────────────────────────────────────────

df = pd.read_csv(MARKSHEET)
print(f"Total patients in marksheet: {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

missing_any = []
missing_all = []
found = []
per_sequence_missing = {seq: 0 for seq in SEQUENCES}

for _, row in df.iterrows():
    patient_id = str(row["patient_id"])
    study_id   = str(row["study_id"])

    # Find which fold this patient is in
    patient_fold = None
    for fold in FOLDS:
        fold_path = IMAGE_ROOT / fold / patient_id
        if fold_path.exists():
            patient_fold = fold_path
            break

    if patient_fold is None:
        missing_all.append(patient_id)
        for seq in SEQUENCES:
            per_sequence_missing[seq] += 1
        continue

    # Check each sequence file
    seq_missing = []
    for seq in SEQUENCES:
        fname = f"{patient_id}_{study_id}_{seq}.mha"
        fpath = patient_fold / fname
        if not fpath.exists():
            seq_missing.append(seq)
            per_sequence_missing[seq] += 1

    if seq_missing:
        missing_any.append({"patient_id": patient_id, "missing_sequences": seq_missing})
    else:
        found.append(patient_id)

# ── REPORT ────────────────────────────────────────────────
print("=" * 60)
print("MISSING FILE AUDIT REPORT")
print("=" * 60)
print(f"  ✅ Fully present : {len(found)}")
print(f"  ⚠️  Missing some  : {len(missing_any)}")
print(f"  ❌ Folder missing : {len(missing_all)}")
print(f"  Total affected   : {len(missing_any) + len(missing_all)}")
print()
print("Missing by sequence:")
for seq, count in per_sequence_missing.items():
    print(f"  {seq}: {count} missing")

print()
print("=" * 60)
print("PATIENTS WITH MISSING FOLDER (first 20):")
print("=" * 60)
for pid in missing_all[:20]:
    print(f"  Patient {pid} — folder not found in any fold")

print()
print("=" * 60)
print("PATIENTS WITH MISSING SEQUENCES (first 20):")
print("=" * 60)
for entry in missing_any[:20]:
    print(f"  Patient {entry['patient_id']} — missing: {entry['missing_sequences']}")

# ── SAVE RESULTS ──────────────────────────────────────────
results_df = pd.DataFrame([
    {"patient_id": pid, "issue": "folder_missing", "missing_sequences": "all"}
    for pid in missing_all
] + [
    {"patient_id": e["patient_id"], "issue": "sequence_missing", "missing_sequences": str(e["missing_sequences"])}
    for e in missing_any
])

out_path = Path("missing_files_audit.csv")
results_df.to_csv(out_path, index=False)
print(f"\n✅ Full audit saved to: {out_path}")
print(f"   Run: cat {out_path} to inspect")
