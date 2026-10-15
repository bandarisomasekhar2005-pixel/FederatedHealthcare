"""
download_dataset.py
--------------------
Downloads the official UCI Heart Disease dataset (Cleveland subset) directly
from the UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
Output directory: data/raw/
"""

import os
import sys
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Official UCI Repository endpoints
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
UCI_DIRECT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
UCI_NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names"

# UCI Cleveland 14 attributes according to heart-disease.names documentation
COLUMN_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male; 0 = female
    "cp",        # chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal, 4: asymptomatic)
    "trestbps",  # resting blood pressure (in mm Hg)
    "chol",      # serum cholestoral in mg/dl
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",   # resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1 = yes; 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
    "ca",        # number of major vessels (0-3) colored by flourosopy
    "thal",      # defect status: 3 = normal; 6 = fixed defect; 7 = reversable defect
    "num"        # diagnosis of heart disease (angiographic disease status: 0 = <50% narrowing, 1-4 = >50% narrowing)
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_uci_heart_disease():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_data_file = RAW_DATA_DIR / "processed.cleveland.data"
    raw_names_file = RAW_DATA_DIR / "heart-disease.names"
    raw_csv_file = RAW_DATA_DIR / "cleveland_raw.csv"
    raw_zip_file = RAW_DATA_DIR / "heart_disease_raw.zip"

    print("=" * 70)
    print("STEP 1: DOWNLOADING REAL UCI HEART DISEASE DATASET (CLEVELAND)")
    print("=" * 70)
    print(f"Target repository: https://archive.ics.uci.edu/dataset/45/heart+disease")
    print(f"Output directory : {RAW_DATA_DIR}")

    download_success = False

    # Attempt 1: Official repository ZIP bundle
    try:
        print(f"\n[INFO] Connecting to UCI ZIP endpoint:\n       {UCI_ZIP_URL}")
        response = requests.get(UCI_ZIP_URL, headers=HEADERS, timeout=25)
        response.raise_for_status()

        if len(response.content) == 0:
            raise ValueError("Downloaded ZIP bundle is empty.")

        # Save raw zip
        with open(raw_zip_file, "wb") as f:
            f.write(response.content)
        print(f"[SUCCESS] Saved raw zip archive to {raw_zip_file.name} ({len(response.content)} bytes)")

        # Unpack processed.cleveland.data and heart-disease.names
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            if "processed.cleveland.data" in z.namelist():
                z.extract("processed.cleveland.data", path=RAW_DATA_DIR)
                print(f"[SUCCESS] Extracted 'processed.cleveland.data' from zip.")
            if "heart-disease.names" in z.namelist():
                z.extract("heart-disease.names", path=RAW_DATA_DIR)
                print(f"[SUCCESS] Extracted 'heart-disease.names' documentation.")

        download_success = True

    except Exception as zip_err:
        print(f"[WARNING] Could not fetch official ZIP bundle: {zip_err}")
        print("[INFO] Attempting direct fallback to official raw data file...")

        # Attempt 2: Direct raw file download from UCI machine-learning-databases
        try:
            r_data = requests.get(UCI_DIRECT_DATA_URL, headers=HEADERS, timeout=25)
            r_data.raise_for_status()
            with open(raw_data_file, "wb") as f:
                f.write(r_data.content)

            r_names = requests.get(UCI_NAMES_URL, headers=HEADERS, timeout=25)
            r_names.raise_for_status()
            with open(raw_names_file, "wb") as f:
                f.write(r_names.content)

            print(f"[SUCCESS] Downloaded raw files directly from UCI.")
            download_success = True
        except Exception as direct_err:
            print("\n" + "!" * 70)
            print("[CRITICAL ERROR] Failed to download real dataset from UCI Repository!")
            print(f"Direct error: {direct_err}")
            print("Per project safety requirements, NO fake or synthetic records will be created.")
            print("\nPlease ensure you have an active internet connection or manually place:")
            print(f"  - 'processed.cleveland.data' into {RAW_DATA_DIR}")
            print(f"You can manually download it using curl or browser from:")
            print(f"  {UCI_DIRECT_DATA_URL}")
            print("!" * 70 + "\n")
            sys.exit(1)

    # Verification: check file existence and size
    if not raw_data_file.exists() or raw_data_file.stat().st_size == 0:
        print(f"[ERROR] Raw data file {raw_data_file} is missing or 0 bytes!")
        sys.exit(1)

    file_size_bytes = raw_data_file.stat().st_size
    print(f"\n[VERIFICATION] Raw dataset verified on disk:")
    print(f"  - File path: {raw_data_file}")
    print(f"  - File size: {file_size_bytes} bytes (non-empty)")

    # Load raw data into DataFrame to check dimensions
    # Raw file has missing values marked as '?'
    df_raw = pd.read_csv(raw_data_file, names=COLUMN_NAMES, na_values="?")

    # Save a clean raw CSV with headers for inspection while preserving all '?' / raw content
    df_raw_for_csv = pd.read_csv(raw_data_file, names=COLUMN_NAMES)
    df_raw_for_csv.to_csv(raw_csv_file, index=False)
    print(f"  - Formatted raw CSV saved to: {raw_csv_file}")

    num_rows, num_cols = df_raw.shape
    print("\n[DATASET CHARACTERISTICS]")
    print(f"  - Total Rows    : {num_rows}")
    print(f"  - Total Columns : {num_cols}")
    print(f"  - Column Names  : {', '.join(COLUMN_NAMES)}")
    print(f"  - Missing values: {df_raw.isna().sum().sum()} total across dataset")
    for col, count in df_raw.isna().sum().items():
        if count > 0:
            print(f"      * {col}: {count} missing values")

    print("\n[COMPLETE] Real UCI Cleveland dataset successfully downloaded and verified.")
    return raw_data_file, raw_csv_file


if __name__ == "__main__":
    download_uci_heart_disease()
