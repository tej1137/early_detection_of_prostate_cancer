
'''Function to calculate missing values among PSA, PSAD, and Prostate Volume columns'''

def calculate_psad_psa_prostate_volume(df, psa_col='psa', psad_col= 'psad', vol_col='prostate_volume'):
    df = df.copy()

    for c in [psa_col, psad_col, vol_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    for _ in range(3):
        #psad = psa / volume
        m = df[psad_col].isna() & df[psa_col].notna() & df[vol_col].notna() & (df[vol_col] > 0)
        df.loc[m, psad_col] = df.loc[m, psa_col] / df.loc[m, vol_col]

        #psa = psad * volume
        m = df[psa_col].isna() & df[psad_col].notna() & df[vol_col].notna() & (df[vol_col] > 0)
        df.loc[m, psa_col] = df.loc[m, psad_col] * df.loc[m, vol_col]

        #volume = psa / psad
        m = df[vol_col].isna() & df[psa_col].notna() & df[psad_col].notna() & (df[psad_col] > 0)
        df.loc[m, vol_col] = df.loc[m, psa_col] / df.loc[m, psad_col]

    return df
import os
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#paths to data
train_data_path = r"F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\split_data\train_clinical.csv" # raw split (before preprocessing)
#output path for preprocessed data
output_train_data_path = r"F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\preprocessed_data"
os.makedirs(output_train_data_path, exist_ok=True)

#Defining features and target variables of interest
features = ["patient_id", "psa", "psad", "patient_age", "prostate_volume"]
target = "case_csPCa"  # YES/NO for clinically significant prostate cancer

#loading training data
df = pd.read_csv(train_data_path)
print(df.head())
print(df.columns.to_list())
print(df.shape)
print(df.isnull().sum())

#Separating features and target variable
x = df[features].copy()
y = df[target]

#checking missing values
print(df["case_csPCa"].value_counts())

#Label encoding
df["case_csPCa"] = df["case_csPCa"].map({"NO": 0, "YES": 1})


# count missing among the 3 columns per row
missing_count = df[features].isna().sum(axis=1)

# keep rows with 0 or 1 missing; drop rows with 2 or 3 missing
df = df[missing_count <= 1].copy()
print(missing_count.value_counts())

#calculate missing values among PSA, PSAD, and Prostate Volume
df = calculate_psad_psa_prostate_volume(df, psa_col='psa', psad_col='psad', vol_col='prostate_volume')

#checking missing values after calculation
print(df[features].isna().sum())
print(df["case_csPCa"].value_counts())

StandardScaler = StandardScaler()

xc_sc = StandardScaler.fit_transform(df[features].drop(columns=["patient_id"]))

train_preprocessed = pd.DataFrame(
    xc_sc,
    columns=[f"{col}_sc" for col in features if col != "patient_id"]
)
train_preprocessed["case_csPCa"] = df["case_csPCa"].values

output_csv = os.path.join(output_train_data_path, "preprocessed_train.csv")
train_preprocessed.to_csv(output_csv, index=False)

print("\n✅ Saved:", output_csv)

