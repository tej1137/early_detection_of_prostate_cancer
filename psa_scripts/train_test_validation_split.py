import pandas as pd
from sklearn.model_selection import train_test_split
import os

# loading the data to split.
base_dir = r"F:\MOD002691 - FP\pi_cai_project"
clinical_dir = os.path.join(base_dir, "picai_labels", "clinical_information")

#new directory to save the splits
data_split_dir = os.path.join(clinical_dir, "split_data")

#reading the data
raw_file = pd.read_csv(clinical_dir + r"\marksheet.csv") 

#inspecting the column names
print("X columns:", raw_file.columns.tolist())

#Combine into a single df for easier saving
df = raw_file.copy()
target = df["case_csPCa"]


#Check original distribution
print("Target distribution:\n", target.value_counts())

# 70% train, 30% temp (stratified)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=target,
    random_state=10,
)

# 15% val, 15% test from that 30% (stratified again)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["case_csPCa"],
    random_state=42,
)

#new directory to save the splits
os.makedirs(data_split_dir, exist_ok=True)
print(f"Created: {data_split_dir}")

# Save splits
train_df.to_csv(data_split_dir + r"\train_clinical.csv", index=False)
val_df.to_csv(data_split_dir + r"\val_clinical.csv", index=False)
test_df.to_csv(data_split_dir + r"\test_clinical.csv", index=False)

#check
print("Data split complete:")
print(f"Total rows: {len(df)}")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

print("\nClass counts:")
print("Train:\n", train_df["case_csPCa"].value_counts())
print("Val:\n", val_df["case_csPCa"].value_counts())
print("Test:\n", test_df["case_csPCa"].value_counts())

