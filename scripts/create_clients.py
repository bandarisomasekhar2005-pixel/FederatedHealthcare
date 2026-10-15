"""
create_clients.py
------------------
Partitions ONLY the 80% global training dataset (global_train.csv, ~242 samples)
across four simulated hospital nodes (Hospital 1, Hospital 2, Hospital 3, Hospital 4)
using a reproducible non-IID Dirichlet distribution:
  DIRICHLET_ALPHA = 0.5

CRITICAL ARCHITECTURE RULE:
  The 20% holdout test set (global_test.csv, ~61 samples) is NEVER partitioned
  and NEVER distributed to hospitals.

Outputs:
  data/splits/hospital_1_train.csv
  data/splits/hospital_2_train.csv
  data/splits/hospital_3_train.csv
  data/splits/hospital_4_train.csv
  (also mirrored to data/clients/hospital_1.csv ... hospital_4.csv for compatibility)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
GLOBAL_TRAIN_FILE = SPLITS_DIR / "global_train.csv"
GLOBAL_TEST_FILE = SPLITS_DIR / "global_test.csv"
CLIENTS_DATA_DIR = PROJECT_ROOT / "data" / "clients"

# Configuration
NUM_CLIENTS = 4
DIRICHLET_ALPHA = 0.5
RANDOM_SEED = 42
MIN_SAMPLES_PER_CLASS = 10  # Guarantees sufficient samples per class for local gradient descent


def partition_dirichlet_non_iid(df, num_clients=4, alpha=0.5, seed=42, min_per_class=10):
    """
    Partitions global_train_df indices across clients using Dirichlet distribution.
    Guarantees reproducible, disjoint partitions where every hospital has valid training
    samples from both classes, while introducing significant non-IID label skew.
    """
    rng = np.random.RandomState(seed)
    targets = df["target"].values
    classes = np.unique(targets)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = np.where(targets == c)[0].copy()
        rng.shuffle(idx_c)
        total_class_samples = len(idx_c)

        if total_class_samples < min_per_class * num_clients:
            raise ValueError(
                f"Class {c} has {total_class_samples} training samples, insufficient for "
                f"min_per_class={min_per_class} across {num_clients} clients."
            )

        base_count = min_per_class
        remaining_samples = total_class_samples - (base_count * num_clients)

        # Draw proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        rem_counts = (proportions * remaining_samples).astype(int)

        # Distribute rounding remainder
        rem_diff = remaining_samples - rem_counts.sum()
        for i in range(rem_diff):
            rem_counts[i % num_clients] += 1

        final_counts = base_count + rem_counts

        # Assign disjoint slices to each client
        offset = 0
        for i in range(num_clients):
            allocated_idx = idx_c[offset : offset + final_counts[i]]
            client_indices[i].extend(allocated_idx)
            offset += final_counts[i]

    # Verify partition integrity
    all_assigned = [idx for c_idx in client_indices for idx in c_idx]
    assert len(all_assigned) == len(df), f"Partition count mismatch: {len(all_assigned)} != {len(df)}"
    assert len(set(all_assigned)) == len(df), "Overlap detected among client partitions!"

    return client_indices


def create_hospital_clients():
    print("=" * 80)
    print("STEP 3: PARTITIONING 80% GLOBAL TRAIN SET INTO FOUR NON-IID HOSPITALS")
    print("=" * 80)
    print(f"Configured parameters:")
    print(f"  - Input Data Source        : {GLOBAL_TRAIN_FILE.name} (80% Global Train ONLY)")
    print(f"  - Number of hospital nodes : {NUM_CLIENTS}")
    print(f"  - Dirichlet Alpha parameter: {DIRICHLET_ALPHA}")
    print(f"  - Random Seed              : {RANDOM_SEED}")
    print(f"  - Min Samples per Class    : {MIN_SAMPLES_PER_CLASS}")

    if not GLOBAL_TRAIN_FILE.exists():
        print(f"[ERROR] Global train file not found at {GLOBAL_TRAIN_FILE}")
        print("Please run `python scripts/preprocess_dataset.py` first.")
        sys.exit(1)

    df_train = pd.read_csv(GLOBAL_TRAIN_FILE)
    total_train_samples = len(df_train)
    print(f"[INFO] Loaded global training dataset: {total_train_samples} samples, {df_train.shape[1]} columns")

    # Clean legacy per-hospital test CSVs if any exist
    for legacy_test_f in SPLITS_DIR.glob("hospital_*_test.csv"):
        try:
            legacy_test_f.unlink()
        except Exception:
            pass

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CLIENTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    client_indices = partition_dirichlet_non_iid(
        df_train,
        num_clients=NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA,
        seed=RANDOM_SEED,
        min_per_class=MIN_SAMPLES_PER_CLASS
    )

    hospital_dfs = []
    print("\n" + "-" * 80)
    print("HOSPITAL TRAINING CLIENT DISTRIBUTIONS (NON-IID, DIRICHLET ALPHA = 0.5)")
    print("-" * 80)

    for i in range(NUM_CLIENTS):
        hosp_idx = client_indices[i]
        hosp_df = df_train.iloc[hosp_idx].copy().reset_index(drop=True)

        # Output to data/splits/hospital_i_train.csv
        hosp_train_file = SPLITS_DIR / f"hospital_{i + 1}_train.csv"
        hosp_df.to_csv(hosp_train_file, index=False)

        # Mirror to data/clients/hospital_i.csv
        hosp_compat_file = CLIENTS_DATA_DIR / f"hospital_{i + 1}.csv"
        hosp_df.to_csv(hosp_compat_file, index=False)

        hospital_dfs.append(hosp_df)

        total = len(hosp_df)
        class_0_cnt = int((hosp_df["target"] == 0).sum())
        class_1_cnt = int((hosp_df["target"] == 1).sum())
        pct_0 = class_0_cnt / total * 100
        pct_1 = class_1_cnt / total * 100

        print(f"Hospital {i + 1}:")
        print(f"  - Saved to        : {hosp_train_file.name}")
        print(f"  - Training samples: {total:3d} ({total / total_train_samples:.1%} of global train)")
        print(f"  - Class 0 (No HD) : {class_0_cnt:3d} ({pct_0:5.1f}%)")
        print(f"  - Class 1 (HD)    : {class_1_cnt:3d} ({pct_1:5.1f}%)")
        print(f"  - Feature count   : {hosp_df.shape[1] - 1} input features + 1 target column")
        print(f"  - Test data leak  : ZERO (All samples originate strictly from global_train.csv)")
        print()

    # Verification: sum of samples
    sum_assigned = sum(len(h) for h in hospital_dfs)
    print("-" * 80)
    print(f"Integrity check:")
    print(f"  - Sum of hospital training samples : {sum_assigned}")
    print(f"  - Global training sample count     : {total_train_samples}")
    assert sum_assigned == total_train_samples, f"Sum mismatch: {sum_assigned} != {total_train_samples}"
    print(f"  - Partitions match 100% without data loss or overlap.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    create_hospital_clients()
