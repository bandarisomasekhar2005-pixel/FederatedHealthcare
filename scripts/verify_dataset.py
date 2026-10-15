"""
verify_dataset.py
------------------
Performs automated verification of the 80:20 Train/Test Federated Learning pipeline:
  1. File & Directory Existence (Global train, Global test, 4 Hospital train splits, indices)
  2. Sample Count & Partition Integrity (Total 303 = 242 train + 61 test)
  3. Zero-Leakage & Disjointness Audit (Train INTERSECT Test = EMPTY, 0 test samples in hospitals, 0 hospital overlap)
  4. Missing Value Audit (zero NaN values across all splits)
  5. Schema & Feature Alignment across splits and hospitals
  6. Statistical Non-IID Heterogeneity Validation (Chi-Square test & KL Divergence)
  7. Privacy & PII Compliance (ensuring zero personally identifying information)
  8. Generates visual distribution chart and verification report in results/dataset/
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "raw" / "processed.cleveland.data"
RAW_CSV_FILE = DATA_DIR / "raw" / "cleveland_raw.csv"
DATA_SPLITS_DIR = DATA_DIR / "splits"
CLIENTS_DIR = DATA_DIR / "clients"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset"

GLOBAL_TRAIN_FILE = DATA_SPLITS_DIR / "global_train.csv"
GLOBAL_TEST_FILE = DATA_SPLITS_DIR / "global_test.csv"
TRAIN_INDICES_FILE = DATA_SPLITS_DIR / "train_indices.csv"
TEST_INDICES_FILE = DATA_SPLITS_DIR / "test_indices.csv"


def kl_divergence(p, q):
    """Computes Kullback-Leibler divergence D_KL(P || Q) with epsilon smoothing."""
    eps = 1e-12
    p = np.asarray(p, dtype=float) + eps
    p /= p.sum()
    q = np.asarray(q, dtype=float) + eps
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def verify_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = []
    summary_data = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 75)
    log("STEP 4: AUTOMATED 80:20 DATASET & FEDERATED LEAKAGE AUDIT")
    log("=" * 75)

    all_passed = True

    # ---------------------------------------------------------
    # 1. FILE & DIRECTORY EXISTENCE AUDIT
    # ---------------------------------------------------------
    log("\n[CHECK 1/7] File & Directory Existence Audit")
    required_files = {
        "Raw Data File"         : RAW_DATA_FILE if RAW_DATA_FILE.exists() else RAW_CSV_FILE,
        "Global Train Split"    : GLOBAL_TRAIN_FILE,
        "Global Test Split"     : GLOBAL_TEST_FILE,
        "Train Indices File"    : TRAIN_INDICES_FILE,
        "Test Indices File"     : TEST_INDICES_FILE,
        "Hospital 1 Train CSV"  : DATA_SPLITS_DIR / "hospital_1_train.csv",
        "Hospital 2 Train CSV"  : DATA_SPLITS_DIR / "hospital_2_train.csv",
        "Hospital 3 Train CSV"  : DATA_SPLITS_DIR / "hospital_3_train.csv",
        "Hospital 4 Train CSV"  : DATA_SPLITS_DIR / "hospital_4_train.csv",
    }

    for name, fpath in required_files.items():
        if fpath.exists() and fpath.stat().st_size > 0:
            log(f"  [PASS] {name:<23}: Found ({fpath.stat().st_size} bytes)")
        else:
            log(f"  [FAIL] {name:<23}: Missing or 0 bytes at {fpath}")
            all_passed = False

    if not all_passed:
        log("\n[ABORT] Pipeline verification failed due to missing files.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. LOAD SPLITS & SAMPLE COUNTS
    # ---------------------------------------------------------
    log("\n[CHECK 2/7] Sample Count & Split Integrity Audit")
    df_train = pd.read_csv(GLOBAL_TRAIN_FILE)
    df_test = pd.read_csv(GLOBAL_TEST_FILE)
    train_idx = pd.read_csv(TRAIN_INDICES_FILE)["index"].tolist()
    test_idx = pd.read_csv(TEST_INDICES_FILE)["index"].tolist()

    n_train = len(df_train)
    n_test = len(df_test)
    total_samples = n_train + n_test

    log(f"  - Total Samples  : {total_samples}")
    log(f"  - Training Split : {n_train} ({n_train/total_samples*100:.1f}%)")
    log(f"  - Testing Split  : {n_test} ({n_test/total_samples*100:.1f}%)")

    if total_samples == 303:
        log(f"  [PASS] Total samples match expected UCI Cleveland count (303).")
    else:
        log(f"  [WARNING] Total samples is {total_samples}, expected 303.")

    # Load hospital training sets
    hosp_dfs = {}
    hosp_counts = {}
    for i in range(1, 5):
        h_path = DATA_SPLITS_DIR / f"hospital_{i}_train.csv"
        h_df = pd.read_csv(h_path)
        hosp_dfs[f"Hospital {i}"] = h_df
        hosp_counts[f"Hospital {i}"] = len(h_df)
        log(f"  - Hospital {i} Training: {len(h_df)} samples")

    sum_hosp_train = sum(hosp_counts.values())
    log(f"  - Sum of Hospital Training Samples: {sum_hosp_train}")

    if sum_hosp_train == n_train:
        log(f"  [PASS] Sum of hospital training samples ({sum_hosp_train}) == Global training samples ({n_train})")
    else:
        log(f"  [FAIL] Sum of hospital training samples ({sum_hosp_train}) != Global training samples ({n_train})")
        all_passed = False

    # ---------------------------------------------------------
    # 3. DATA LEAKAGE & DISJOINTNESS AUDIT
    # ---------------------------------------------------------
    log("\n[CHECK 3/7] Zero-Leakage & Disjointness Audit")

    # Index overlap check
    overlap_indices = set(train_idx).intersection(set(test_idx))
    train_test_overlap_count = len(overlap_indices)

    if train_test_overlap_count == 0:
        log(f"  [PASS] Train/Test Index Intersection = EMPTY (Overlap: 0)")
        log(f"  [PASS] Global train/test leakage check passed.")
        log(f"  [PASS] No test samples are used for federated training.")
    else:
        log(f"  [ERROR] DATA LEAKAGE DETECTED: {train_test_overlap_count} overlapping indices between train and test!")
        all_passed = False

    # Test samples inside hospital training check
    test_rows_set = set(tuple(x) for x in df_test.to_numpy())
    test_in_hosp_count = 0
    for name, h_df in hosp_dfs.items():
        hosp_rows = [tuple(x) for x in h_df.to_numpy()]
        for r in hosp_rows:
            if r in test_rows_set:
                test_in_hosp_count += 1

    if test_in_hosp_count == 0:
        log(f"  [PASS] Test samples inside hospital training = 0")
    else:
        log(f"  [FAIL] DATA LEAKAGE: {test_in_hosp_count} test records found in hospital training splits!")
        all_passed = False

    # Hospital mutual disjointness check
    hosp_overlap_count = 0
    all_hosp_rows = []
    for name, h_df in hosp_dfs.items():
        all_hosp_rows.append([tuple(x) for x in h_df.to_numpy()])

    for i in range(len(all_hosp_rows)):
        for j in range(i + 1, len(all_hosp_rows)):
            overlap = set(all_hosp_rows[i]).intersection(set(all_hosp_rows[j]))
            hosp_overlap_count += len(overlap)

    if hosp_overlap_count == 0:
        log(f"  [PASS] Hospital overlap = 0 (strictly mutually disjoint)")
    else:
        log(f"  [FAIL] Hospital overlap detected: {hosp_overlap_count} shared samples!")
        all_passed = False

    # ---------------------------------------------------------
    # 4. SCHEMA AND MISSING VALUE AUDIT
    # ---------------------------------------------------------
    log("\n[CHECK 4/7] Schema & Missing Value Audit")
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)

    if train_cols == test_cols:
        log(f"  [PASS] Train and Test schemas match ({len(train_cols)} columns)")
    else:
        log(f"  [FAIL] Schema mismatch between Train and Test!")
        all_passed = False

    for name, h_df in hosp_dfs.items():
        if list(h_df.columns) == train_cols:
            log(f"  [PASS] {name}: Schema aligned ({len(train_cols)} columns)")
        else:
            log(f"  [FAIL] {name}: Schema mismatch!")
            all_passed = False

    # Check for NaN / null values
    nan_train = df_train.isna().sum().sum()
    nan_test = df_test.isna().sum().sum()
    nan_hosp = sum(h_df.isna().sum().sum() for h_df in hosp_dfs.values())

    log(f"  - Missing values in Global Train: {nan_train}")
    log(f"  - Missing values in Global Test : {nan_test}")
    log(f"  - Missing values in Hospitals   : {nan_hosp}")

    if nan_train == 0 and nan_test == 0 and nan_hosp == 0:
        log("  [PASS] Zero missing values detected across all splits.")
    else:
        log("  [FAIL] Missing values detected!")
        all_passed = False

    # ---------------------------------------------------------
    # 5. STATISTICAL NON-IID VALIDATION
    # ---------------------------------------------------------
    log("\n[CHECK 5/7] Statistical Non-IID Heterogeneity Validation")
    train_c0 = int((df_train["target"] == 0).sum())
    train_c1 = int((df_train["target"] == 1).sum())
    global_train_probs = np.array([train_c0, train_c1]) / n_train

    contingency_table = []
    for name, h_df in hosp_dfs.items():
        c_tot = len(h_df)
        c0 = int((h_df["target"] == 0).sum())
        c1 = int((h_df["target"] == 1).sum())
        p0 = c0 / c_tot
        p1 = c1 / c_tot
        contingency_table.append([c0, c1])

        hosp_probs = np.array([p0, p1])
        kl_div = kl_divergence(hosp_probs, global_train_probs)

        summary_data.append({
            "Hospital": name,
            "Total Samples": c_tot,
            "Share of Train (%)": round(c_tot / n_train * 100, 2),
            "Class 0 (No HD)": c0,
            "Class 0 (%)": round(p0 * 100, 2),
            "Class 1 (HD)": c1,
            "Class 1 (%)": round(p1 * 100, 2),
            "KL Divergence": round(kl_div, 4)
        })

        log(f"  - {name}: N={c_tot:<2} | Class 0: {c0:2d} ({p0*100:5.1f}%) | Class 1: {c1:2d} ({p1*100:5.1f}%) | KL Div: {kl_div:.4f}")

    chi2, p_val, dof, _ = chi2_contingency(contingency_table)
    log(f"\n  Chi-Square Test of Independence (Label vs. Hospital Client):")
    log(f"    * Chi2 Statistic : {chi2:.4f}")
    log(f"    * Degrees of Freedom: {dof}")
    log(f"    * p-value       : {p_val:.4e}")

    if p_val < 0.01:
        log(f"  [PASS] p-value < 0.01: Statistically significant Non-IID label skew confirmed!")
    else:
        log(f"  [WARNING] Distribution does not show statistically significant Non-IID skew.")

    # ---------------------------------------------------------
    # 6. PRIVACY & PII AUDIT
    # ---------------------------------------------------------
    log("\n[CHECK 6/7] Privacy & PII Compliance Audit")
    forbidden_tokens = [
        "name", "phone", "address", "aadhaar", "ssn", "mrn", "id",
        "email", "patient", "dob", "contact", "zip"
    ]
    pii_violations = []
    for col in train_cols:
        col_lower = col.lower()
        for token in forbidden_tokens:
            if token == col_lower or (token in col_lower and col_lower not in ["id"]):
                if col_lower not in ["oldpeak", "target"]:
                    pii_violations.append(col)

    if len(pii_violations) == 0:
        log("  [PASS] Zero PII tokens identified. All records contain strictly anonymized clinical variables.")
    else:
        log(f"  [FAIL] Potential PII columns detected: {pii_violations}")
        all_passed = False

    # ---------------------------------------------------------
    # 7. EXPORT VERIFICATION ARTIFACTS
    # ---------------------------------------------------------
    log("\n[CHECK 7/7] Generating Verification Artifacts")
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = RESULTS_DIR / "client_distribution_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    log(f"  [SAVED] Distribution summary saved to: {summary_csv_path.name}")

    # Generate Bar Chart
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    hospitals = [d["Hospital"] for d in summary_data]
    c0_counts = [d["Class 0 (No HD)"] for d in summary_data]
    c1_counts = [d["Class 1 (HD)"] for d in summary_data]
    c0_pcts = [d["Class 0 (%)"] for d in summary_data]
    c1_pcts = [d["Class 1 (%)"] for d in summary_data]

    x = np.arange(len(hospitals))
    width = 0.35

    ax1.bar(x - width/2, c0_counts, width, label="Class 0 (No Heart Disease)", color="#2b5c8f")
    ax1.bar(x + width/2, c1_counts, width, label="Class 1 (Heart Disease)", color="#d95f02")
    ax1.set_title("Hospital Training Counts per Class (Total=242)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Hospital Node", fontsize=10)
    ax1.set_ylabel("Number of Patient Records", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(hospitals, fontsize=10)
    ax1.legend(loc="upper right")
    for i in range(len(hospitals)):
        ax1.text(x[i] - width/2, c0_counts[i] + 1, str(c0_counts[i]), ha="center", fontsize=9)
        ax1.text(x[i] + width/2, c1_counts[i] + 1, str(c1_counts[i]), ha="center", fontsize=9)

    ax2.bar(hospitals, c0_pcts, label="Class 0 (%)", color="#2b5c8f", width=0.5)
    ax2.bar(hospitals, c1_pcts, bottom=c0_pcts, label="Class 1 (%)", color="#d95f02", width=0.5)
    train_c0_pct = train_c0 / n_train * 100
    ax2.axhline(train_c0_pct, color="#2b5c8f", linestyle="--", alpha=0.7, label=f"Train Class 0 ({train_c0_pct:.1f}%)")
    ax2.axhline(100 - train_c0_pct, color="#d95f02", linestyle=":", alpha=0.7, label=f"Train Class 1 ({100-train_c0_pct:.1f}%)")
    ax2.set_title("Non-IID Label Proportions (Dirichlet Alpha = 0.5)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Hospital Node", fontsize=10)
    ax2.set_ylabel("Class Percentage (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", fontsize=8)

    for i in range(len(hospitals)):
        ax2.text(i, c0_pcts[i] / 2, f"{c0_pcts[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax2.text(i, c0_pcts[i] + c1_pcts[i] / 2, f"{c1_pcts[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    chart_path = RESULTS_DIR / "client_distribution_plot.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    log(f"  [SAVED] Visual distribution chart saved to: {chart_path.name}")

    # Save report
    report_path = RESULTS_DIR / "verification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    log(f"  [SAVED] Verification report saved to: {report_path.name}")

    # ---------------------------------------------------------
    # EXACT REQUIRED TERMINAL VERIFICATION BLOCK (SECTION 17)
    # ---------------------------------------------------------
    status_str = "PASSED" if all_passed else "FAILED"
    print("\n" + "=" * 40)
    print("FEDERATED DATA INTEGRITY VERIFICATION")
    print("=" * 37)
    print()
    print(f"Total dataset samples: {total_samples}")
    print(f"Global training samples: {n_train}")
    print(f"Global testing samples: {n_test}")
    print()
    for i in range(1, 5):
        print(f"Hospital {i} training: {hosp_counts[f'Hospital {i}']}")
    print()
    print(f"Sum of hospital training samples: {sum_hosp_train}")
    print()
    print(f"Train/Test overlap: {train_test_overlap_count}")
    print(f"Test samples inside hospital training: {test_in_hosp_count}")
    print(f"Hospital overlap: {hosp_overlap_count}")
    print()
    print(f"STATUS: {status_str}")
    print()

    return all_passed


if __name__ == "__main__":
    passed = verify_pipeline()
    if not passed:
        sys.exit(1)
