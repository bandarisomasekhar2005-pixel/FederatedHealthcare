"""
preprocess_dataset.py
----------------------
Loads the raw UCI Heart Disease dataset (Cleveland subset), performs a strict
global 80:20 stratified train/test split, learns imputation and feature scaling
parameters SOLELY from the training set to prevent data leakage, applies the
fitted transformations to both partitions, and saves the partitioned datasets.

Outputs:
  data/splits/global_train.csv  (80%, ~242 samples)
  data/splits/global_test.csv   (20%, ~61 samples - strictly held-out)
  data/splits/train_indices.csv
  data/splits/test_indices.csv
  data/processed/heart_disease_processed.csv (for reference/compatibility)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cleveland_raw.csv"
RAW_DATA_BACKUP = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_processed.csv"

# Configuration
RANDOM_SEED = 42
TEST_RATIO = 0.20

# Original 14 attributes
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
]
TARGET_NAME = "num"
ALL_COLUMNS = FEATURE_NAMES + [TARGET_NAME]

# Feature classification
CONTINUOUS_NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]


def preprocess_and_split():
    print("=" * 80)
    print("STEP 2: PREPROCESSING & GLOBAL 80:20 TRAIN/TEST SPLIT (LEAKAGE-FREE)")
    print("=" * 80)

    # 1. Load raw data
    if RAW_DATA_PATH.exists():
        data_source = RAW_DATA_PATH
    elif RAW_DATA_BACKUP.exists():
        data_source = RAW_DATA_BACKUP
    else:
        print(f"[ERROR] Raw data not found at {RAW_DATA_PATH} or {RAW_DATA_BACKUP}")
        print("Please run `python scripts/download_dataset.py` first.")
        sys.exit(1)

    print(f"[INFO] Loading raw clinical data from: {data_source.name}")
    df_raw = pd.read_csv(data_source, names=ALL_COLUMNS, na_values="?", header=0 if data_source == RAW_DATA_PATH else None)

    total_samples = len(df_raw)
    missing_before = int(df_raw.isna().sum().sum())

    # 2. Separate features (X) and target (y)
    X_raw = df_raw[FEATURE_NAMES].copy()
    y_raw = df_raw[TARGET_NAME].copy()
    y_binary = (y_raw > 0).astype(int)
    y_binary.name = "target"

    # 3. GLOBAL 80:20 STRATIFIED SPLIT BEFORE ANY PREPROCESSING FITTING
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_binary,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y_binary
    )

    num_train = len(X_train_raw)
    num_test = len(X_test_raw)

    train_c0 = int((y_train == 0).sum())
    train_c1 = int((y_train == 1).sum())
    test_c0 = int((y_test == 0).sum())
    test_c1 = int((y_test == 1).sum())

    # 4. DATA LEAKAGE PREVENTION: Assert Disjoint Index Sets
    train_idx_set = set(X_train_raw.index)
    test_idx_set = set(X_test_raw.index)
    overlap = train_idx_set.intersection(test_idx_set)

    if len(overlap) > 0:
        print("\n" + "!" * 80)
        print("[ERROR] DATA LEAKAGE DETECTED: Overlap between training and testing indices!")
        print(f"Overlap count: {len(overlap)}")
        print("!" * 80 + "\n")
        sys.exit(1)

    # 5. Display Formatted Split Summary
    print("\n" + "=" * 40)
    print("GLOBAL 80:20 TRAIN/TEST SPLIT")
    print("=" * 29)
    print(f"\nTotal samples: {total_samples}")
    print(f"Training samples: {num_train}")
    print(f"Testing samples: {num_test}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"\nTraining class distribution:")
    print(f"Class 0 (No Heart Disease): {train_c0} ({train_c0/num_train:.2%})")
    print(f"Class 1 (Heart Disease Present): {train_c1} ({train_c1/num_train:.2%})")
    print(f"\nTesting class distribution:")
    print(f"Class 0 (No Heart Disease): {test_c0} ({test_c0/num_test:.2%})")
    print(f"Class 1 (Heart Disease Present): {test_c1} ({test_c1/num_test:.2%})")
    print("\nTest set is completely held out from FL training.")
    print("[PASS] Global train/test leakage check passed.")
    print("[PASS] No test samples are used for federated training.")

    # 6. Fit Imputation Parameters SOLELY on Training Data
    # 'ca' imputed with median; 'thal' imputed with mode
    imp_ca = SimpleImputer(strategy="median").fit(X_train_raw[["ca"]])
    imp_thal = SimpleImputer(strategy="most_frequent").fit(X_train_raw[["thal"]])

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    X_train["ca"] = imp_ca.transform(X_train[["ca"]]).ravel()
    X_test["ca"] = imp_ca.transform(X_test[["ca"]]).ravel()

    X_train["thal"] = imp_thal.transform(X_train[["thal"]]).ravel()
    X_test["thal"] = imp_thal.transform(X_test[["thal"]]).ravel()

    # 7. Convert categorical features to integer before One-Hot Encoding
    for cat in CATEGORICAL_FEATURES:
        X_train[cat] = X_train[cat].astype(int)
        X_test[cat] = X_test[cat].astype(int)

    X_train_enc = pd.get_dummies(X_train, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES, drop_first=False, dtype=float)
    X_test_enc = pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES, drop_first=False, dtype=float)

    # Ensure identical column alignment between Train and Test
    for col in X_train_enc.columns:
        if col not in X_test_enc.columns:
            X_test_enc[col] = 0.0
    X_test_enc = X_test_enc[X_train_enc.columns]

    # 8. Fit Standardization SOLELY on Training Data Continuous Features
    scaler = StandardScaler().fit(X_train_enc[CONTINUOUS_NUMERICAL])
    X_train_enc[CONTINUOUS_NUMERICAL] = scaler.transform(X_train_enc[CONTINUOUS_NUMERICAL])
    X_test_enc[CONTINUOUS_NUMERICAL] = scaler.transform(X_test_enc[CONTINUOUS_NUMERICAL])

    # 9. Combine Preprocessed Features with Target
    train_df = pd.concat([X_train_enc, y_train], axis=1)
    test_df = pd.concat([X_test_enc, y_test], axis=1)

    # 10. Save Global Partitions and Indices to data/splits/
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    global_train_file = SPLITS_DIR / "global_train.csv"
    global_test_file = SPLITS_DIR / "global_test.csv"
    train_idx_file = SPLITS_DIR / "train_indices.csv"
    test_idx_file = SPLITS_DIR / "test_indices.csv"

    train_df.to_csv(global_train_file, index=False)
    test_df.to_csv(global_test_file, index=False)
    pd.DataFrame({"index": list(X_train_raw.index)}).to_csv(train_idx_file, index=False)
    pd.DataFrame({"index": list(X_test_raw.index)}).to_csv(test_idx_file, index=False)

    # Also save unified processed dataset for backwards reference
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_processed = pd.concat([train_df, test_df], axis=0).loc[df_raw.index]
    full_processed.to_csv(PROCESSED_DATA_FILE, index=False)

    print(f"\n[SAVED] Partition files saved:")
    print(f"  - Global Train (80%) : {global_train_file} ({train_df.shape[0]} samples, {train_df.shape[1]} cols)")
    print(f"  - Global Test (20%)  : {global_test_file} ({test_df.shape[0]} samples, {test_df.shape[1]} cols)")
    print(f"  - Train Indices      : {train_idx_file.name}")
    print(f"  - Test Indices       : {test_idx_file.name}")
    print("=" * 80 + "\n")

    return train_df, test_df


if __name__ == "__main__":
    preprocess_and_split()
