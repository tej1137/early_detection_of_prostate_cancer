
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

'''Main preprocessing function : can be executed through command line for each  of the split data'''
def preprocess_file(input_csv, output_csv):

    #Defining features and target variables of interest
    features = ["patient_id", "psa", "psad", "patient_age", "prostate_volume"]
    target = "case_csPCa"  # YES/NO for clinically significant prostate cancer

    #loading training data
    df = pd.read_csv(input_csv)

    print(df.head())
    print(df.columns.to_list())
    print(df.shape)
    print(df.isnull().sum())

    #Separating features and target variable
    x = df[features].copy()
    y = df[target]

    #checking missing values
    print(df[target].value_counts())

    #Label encoding
    df[target] = df[target].map({"NO": 0, "YES": 1})

    # count missing among the 3 columns per row
    missing_count = df[features].isna().sum(axis=1)

    # keep rows with 0 or 1 missing; drop rows with 2 or 3 missing
    df = df[missing_count <= 1].copy()
    print(missing_count.value_counts())

    #calculate missing values among PSA, PSAD, and Prostate Volume
    df = calculate_psad_psa_prostate_volume(df, psa_col='psa', psad_col='psad', vol_col='prostate_volume')

    #checking missing values after calculation
    print(df[features].isna().sum())
    print(df[target].value_counts())

    model_features = ["psa", "psad", "patient_age", "prostate_volume"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[model_features])


    train_preprocessed = pd.DataFrame(
        X_scaled,columns=[f"{col}_sc" for col in features if col != "patient_id"])
    
    train_preprocessed[target] = df[target].values

    out = pd.DataFrame(X_scaled, columns=[f"{c}_sc" for c in model_features])
    out[target] = df[target].astype(int).values

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("✅ Saved:", output_csv)
    print("Shape:", out.shape)
    print("Label distribution:\n", out[target].value_counts())

def main():
    parser = argparse.ArgumentParser(description="Preprocess PI-CAI clinical CSV.")  # argparse standard usage [web:366]
    parser.add_argument("input_csv", help="Path to input raw split CSV")
    parser.add_argument("output_csv", help="Path to output preprocessed CSV")
    args = parser.parse_args()  # argparse parses CLI args [web:364]

    preprocess_file(args.input_csv, args.output_csv)



if __name__ == "__main__":
    main()

#Training example command:
# python baseline_model\preprocess_clinical.py "F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\split_data\train_clinical.csv" `
#  "F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\preprocessed_data\preprocessed_train.csv"
#
#validation example command:
# python baseline_model\preprocess_clinical.py "F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\split_data\val_clinical.csv", "F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\preprocessed_data\preprocessed_val.csv"
#
#test example command:
#python baseline_model\preprocess_clinical.py "F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\split_data\test_clinical.csv" `
# "F:\MOD002691 - FP\pi_cai_project\picai_labels\clinical_information\preprocessed_data\preprocessed_test.csv"

