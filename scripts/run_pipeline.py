"""
run_pipeline.py
---------------
End-to-end execution script for the dataset pipeline of:
"Privacy-Preserving Federated Learning for Healthcare"

Runs sequentially:
  1. scripts/download_dataset.py
  2. scripts/preprocess_dataset.py
  3. scripts/create_clients.py
  4. scripts/verify_dataset.py
"""

import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

STEPS = [
    ("Step 1: Download Real UCI Cleveland Dataset", SCRIPT_DIR / "download_dataset.py"),
    ("Step 2: Preprocess Dataset & Feature Pipeline", SCRIPT_DIR / "preprocess_dataset.py"),
    ("Step 3: Partition into 4 Non-IID Hospital Clients", SCRIPT_DIR / "create_clients.py"),
    ("Step 4: Automated Verification & Result Generation", SCRIPT_DIR / "verify_dataset.py"),
]


def run_pipeline():
    print("=" * 80)
    print("PRIVACY-PRESERVING FEDERATED LEARNING FOR HEALTHCARE")
    print("DATASET PIPELINE ORCHESTRATION")
    print("=" * 80)

    for desc, script_path in STEPS:
        print(f"\n>>> Running {desc}...")
        res = subprocess.run([PYTHON_EXE, str(script_path)], check=False)
        if res.returncode != 0:
            print(f"\n[PIPELINE ABORTED] Error encountered while running: {script_path.name}")
            sys.exit(res.returncode)

    print("\n" + "=" * 80)
    print("DATASET PIPELINE SUCCESSFULLY COMPLETED!")
    print("=" * 80)
    print("Artifacts generated:")
    print("  - Raw data      : data/raw/processed.cleveland.data")
    print("  - Processed CSV : data/processed/heart_disease_processed.csv")
    print("  - Hospital nodes: data/clients/hospital_1.csv ... hospital_4.csv")
    print("  - Results       : results/dataset/verification_report.txt")
    print("                    results/dataset/client_distribution_summary.csv")
    print("                    results/dataset/client_distribution_plot.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_pipeline()
