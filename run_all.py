"""
run_all.py
----------
Master runner executing the end-to-end workflow for:
"Privacy-Preserving Federated Learning for Healthcare (80:20 Train/Test Methodology)"

Pipeline Architecture:
  303 Total UCI Cleveland Samples
  |
  +--- 80% GLOBAL TRAINING DATA (242 samples, zero test leakage)
  |          |
  |          +--> Distributed across 4 Non-IID Hospital Clients (Dirichlet alpha = 0.5)
  |               |
  |               +--> Hospital 1 Local Training (84 samples)
  |               +--> Hospital 2 Local Training (33 samples)
  |               +--> Hospital 3 Local Training (36 samples)
  |               +--> Hospital 4 Local Training (89 samples)
  |                         |
  |                         v
  |                  Federated Aggregation
  |                  (FedAvg / FedProx / FedAvg+DP / FedProx+DP)
  |                         |
  |                         v
  |                  FINAL GLOBAL MODELS
  |
  +--- 20% GLOBAL TESTING DATA (61 samples, completely held out)
             |
             v
      FINAL CLINICAL EVALUATION ONLY ON GLOBAL FL MODELS

Sequence:
  [1] Download Official UCI Dataset                          (scripts/download_dataset.py)
  [2] Clinical Preprocessing & Global 80:20 Train/Test Split (scripts/preprocess_dataset.py)
  [3] Partition ONLY Global Train Set to 4 Non-IID Hospitals (scripts/create_clients.py)
  [4] Automated Data Integrity & Zero-Leakage Audit          (scripts/verify_dataset.py)
  [5] Federated Learning & Final 20% Holdout Evaluation      (scripts/train_federated.py)
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON_EXE = sys.executable

PIPELINE_STEPS = [
    ("Stage 1: Downloading Official UCI Cleveland Dataset", SCRIPTS_DIR / "download_dataset.py"),
    ("Stage 2: Clinical Feature Preprocessing & Global 80:20 Split", SCRIPTS_DIR / "preprocess_dataset.py"),
    ("Stage 3: Partitioning ONLY 80% Train Set to 4 Non-IID Hospitals", SCRIPTS_DIR / "create_clients.py"),
    ("Stage 4: Automated Data Integrity & Zero-Leakage Audit", SCRIPTS_DIR / "verify_dataset.py"),
    ("Stage 5: Federated Training & Final Holdout Evaluation", SCRIPTS_DIR / "train_federated.py"),
]


def main():
    print("=" * 85)
    print(" PRIVACY-PRESERVING FEDERATED LEARNING FOR HEALTHCARE")
    print(" MASTER EXECUTION PIPELINE (80:20 HOLDOUT METHODOLOGY)")
    print("=" * 85)

    for step_num, (title, script_path) in enumerate(PIPELINE_STEPS, 1):
        print(f"\n[{step_num}/{len(PIPELINE_STEPS)}] {title}")
        print("-" * 85)
        res = subprocess.run([PYTHON_EXE, str(script_path)], cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            print(f"\n[FAILED] Script {script_path.name} failed with exit code {res.returncode}. Aborting.")
            sys.exit(res.returncode)

    print("\n" + "=" * 85)
    print(" ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 85)
    print("\nKey Artifacts & Deliverables Generated:")
    print("  - Raw Dataset       : data/raw/cleveland_raw.csv")
    print("  - Global Splits     : data/splits/global_train.csv (242 samples, 80%)")
    print("                        data/splits/global_test.csv  (61 samples, 20%)")
    print("                        data/splits/train_indices.csv, test_indices.csv")
    print("  - Hospital Clients  : data/splits/hospital_1_train.csv ... hospital_4_train.csv")
    print("  - Dataset Audit     : results/dataset/verification_report.txt")
    print("                        results/dataset/client_distribution_plot.png")
    print("                        results/dataset/client_distribution_summary.csv")
    print("  - FL Holdout Results: results/federated/holdout_80_20/final_test_metrics.csv")
    print("                        results/federated/holdout_80_20/FedAvg_final_confusion_matrix.png")
    print("                        results/federated/holdout_80_20/FedProx_final_confusion_matrix.png")
    print("                        results/federated/holdout_80_20/FedAvg_DP_final_confusion_matrix.png")
    print("                        results/federated/holdout_80_20/FedProx_DP_final_confusion_matrix.png")
    print("                        results/federated/holdout_80_20/fl_algorithms_roc_curves.png")
    print("                        results/federated/holdout_80_20/fl_training_convergence.png")
    print("                        results/federated/holdout_80_20/model_comparison_bar.png")
    print("                        results/federated/holdout_80_20/federated_80_20_test_report.txt")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
