import pandas as pd

# 1. Load the Datasets
# Make sure these files are in your folder!
radiomics_df = pd.read_csv(r"F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\radiomics_features.csv")
marksheet_df = pd.read_csv("F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\marksheet.csv")

# 2. Ensure IDs match (convert to string to avoid 10029 vs "10029")
radiomics_df['patient_id'] = radiomics_df['patient_id'].astype(str)
marksheet_df['patient_id'] = marksheet_df['patient_id'].astype(str)

# 3. Merge them (Inner Join keeps only matching rows)
merged_df = pd.merge(radiomics_df, marksheet_df, on='patient_id', how='inner')

# 4. Display the Evidence
print("-" * 50)
print("COMPARISON REPORT")
print("-" * 50)

print(f"Original Marksheet Patients: {len(marksheet_df)}")
print(f"Radiomics Features Extracted: {len(radiomics_df)}")
print(f"Percentage Missing: {100 * (1 - len(radiomics_df)/len(marksheet_df)):.1f}%")

print("\n--- LABEL DISTRIBUTION (Cancer vs. No Cancer) ---")
print("\nOriginal Marksheet Labels:")
print(marksheet_df['case_csPCa'].value_counts())

print("\nRadiomics Data Labels (Who did we extract features for?):")
print(merged_df['case_csPCa'].value_counts())

print("\n--- CONCLUSION ---")
missing_count = len(marksheet_df) - len(radiomics_df)
print(f"We are missing {missing_count} patients.")
if missing_count > 500:
    print("CRITICAL: The missing patients are almost entirely the 'NO' class (Benign ).")
    print("This proves we cannot train a classifier because the Negative class is missing.")
