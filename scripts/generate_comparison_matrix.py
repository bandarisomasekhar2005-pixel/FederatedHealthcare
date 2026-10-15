"""
generate_comparison_matrix.py
------------------------------
Generates a side-by-side Comparative Evaluation Matrix and visual comparison charts
between:
  1. Initial Results (Isolated Hospital Models, FedAvg, FedProx, FedAvg + DP)
  2. Current Results (Traditional Centralized 80:20 Train/Test ML Evaluation)

Exports:
  - results/traditional_testing/initial_vs_current_comparison_matrix.csv
  - results/traditional_testing/initial_vs_current_matrix_chart.png
  - results/traditional_testing/confusion_matrices_comparison.png
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "traditional_testing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

matrix_rows = [
    {
        "Evaluation Stage": "Initial Baseline (Siloed)",
        "Model / Method": "Hospital 1 Standalone",
        "Data Distribution": "Non-IID (Hospital 1)",
        "Train Samples": 86,
        "Test Samples": 62,
        "Accuracy (%)": 74.19,
        "Precision (%)": 70.00,
        "Recall / Sens (%)": 75.00,
        "Specificity (%)": 73.53,
        "F1-Score": 0.7241,
        "ROC-AUC": 0.7889,
        "Confusion Matrix [TN,FP,FN,TP]": "[25, 9, 7, 21]",
        "Privacy Protection": "Local Isolation Only"
    },
    {
        "Evaluation Stage": "Initial Baseline (Siloed)",
        "Model / Method": "Hospital 2 Standalone",
        "Data Distribution": "Non-IID (Hospital 2, 76.5% HD)",
        "Train Samples": 40,
        "Test Samples": 62,
        "Accuracy (%)": 67.74,
        "Precision (%)": 59.09,
        "Recall / Sens (%)": 92.86,
        "Specificity (%)": 47.06,
        "F1-Score": 0.7222,
        "ROC-AUC": 0.7920,
        "Confusion Matrix [TN,FP,FN,TP]": "[16, 18, 2, 26]",
        "Privacy Protection": "Local Isolation Only"
    },
    {
        "Evaluation Stage": "Initial Baseline (Siloed)",
        "Model / Method": "Hospital 3 Standalone",
        "Data Distribution": "Non-IID (Hospital 3, 59.2% HD)",
        "Train Samples": 39,
        "Test Samples": 62,
        "Accuracy (%)": 75.81,
        "Precision (%)": 68.57,
        "Recall / Sens (%)": 85.71,
        "Specificity (%)": 67.65,
        "F1-Score": 0.7619,
        "ROC-AUC": 0.8183,
        "Confusion Matrix [TN,FP,FN,TP]": "[23, 11, 4, 24]",
        "Privacy Protection": "Local Isolation Only"
    },
    {
        "Evaluation Stage": "Initial Baseline (Siloed)",
        "Model / Method": "Hospital 4 Standalone",
        "Data Distribution": "Non-IID (Hospital 4, 26.3% HD)",
        "Train Samples": 76,
        "Test Samples": 62,
        "Accuracy (%)": 79.03,
        "Precision (%)": 89.47,
        "Recall / Sens (%)": 60.71,
        "Specificity (%)": 94.12,
        "F1-Score": 0.7234,
        "ROC-AUC": 0.8456,
        "Confusion Matrix [TN,FP,FN,TP]": "[32, 2, 11, 17]",
        "Privacy Protection": "Local Isolation Only"
    },
    {
        "Evaluation Stage": "Initial Collaborative FL",
        "Model / Method": "FedAvg (McMahan et al.)",
        "Data Distribution": "Distributed Non-IID (4 Hospitals)",
        "Train Samples": 241,
        "Test Samples": 62,
        "Accuracy (%)": 74.19,
        "Precision (%)": 71.43,
        "Recall / Sens (%)": 71.43,
        "Specificity (%)": 76.47,
        "F1-Score": 0.7143,
        "ROC-AUC": 0.8298,
        "Confusion Matrix [TN,FP,FN,TP]": "[26, 8, 8, 20]",
        "Privacy Protection": "Weights Shared (Zero Raw Data)"
    },
    {
        "Evaluation Stage": "Initial Collaborative FL",
        "Model / Method": "FedProx (mu=0.01)",
        "Data Distribution": "Distributed Non-IID (4 Hospitals)",
        "Train Samples": 241,
        "Test Samples": 62,
        "Accuracy (%)": 74.19,
        "Precision (%)": 68.75,
        "Recall / Sens (%)": 78.57,
        "Specificity (%)": 70.59,
        "F1-Score": 0.7333,
        "ROC-AUC": 0.8298,
        "Confusion Matrix [TN,FP,FN,TP]": "[24, 10, 6, 22]",
        "Privacy Protection": "Weights Shared (Zero Raw Data)"
    },
    {
        "Evaluation Stage": "Initial Collaborative FL",
        "Model / Method": "FedAvg + DP (Differential Privacy)",
        "Data Distribution": "Distributed Non-IID (4 Hospitals)",
        "Train Samples": 241,
        "Test Samples": 62,
        "Accuracy (%)": 79.03,
        "Precision (%)": 74.19,
        "Recall / Sens (%)": 82.14,
        "Specificity (%)": 76.47,
        "F1-Score": 0.7797,
        "ROC-AUC": 0.8403,
        "Confusion Matrix [TN,FP,FN,TP]": "[26, 8, 5, 23]",
        "Privacy Protection": "Local DP (Gaussian Noise, eps~86)"
    },
    {
        "Evaluation Stage": "Current Benchmark (Centralized)",
        "Model / Method": "Traditional Centralized 80:20 (HeartDiseaseMLP)",
        "Data Distribution": "Pooled Centralized (80:20 Split)",
        "Train Samples": 242,
        "Test Samples": 61,
        "Accuracy (%)": 83.61,
        "Precision (%)": 78.12,
        "Recall / Sens (%)": 89.29,
        "Specificity (%)": 78.79,
        "F1-Score": 0.8333,
        "ROC-AUC": 0.9102,
        "Confusion Matrix [TN,FP,FN,TP]": "[26, 7, 3, 25]",
        "Privacy Protection": "None (Raw Data Centrally Pooled)"
    }
]

def main():
    df_matrix = pd.DataFrame(matrix_rows)
    csv_matrix_path = OUT_DIR / "initial_vs_current_comparison_matrix.csv"
    df_matrix.to_csv(csv_matrix_path, index=False)
    print(f"[SAVED] Master Comparison Matrix CSV: {csv_matrix_path}")

    # Display in terminal
    print("\n" + "=" * 110)
    print("MASTER EVALUATION MATRIX: INITIAL RESULTS vs. CURRENT TRADITIONAL 80:20")
    print("=" * 110)
    print(df_matrix[["Evaluation Stage", "Model / Method", "Accuracy (%)", "Precision (%)", "Recall / Sens (%)", "F1-Score", "ROC-AUC", "Confusion Matrix [TN,FP,FN,TP]"]].to_string(index=False))
    print("=" * 110)

    # 1. Visualization: Multi-metric Matrix Heatmap / Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    models = [
        "Hosp 1\n(Initial)", "Hosp 2\n(Initial)", "Hosp 3\n(Initial)", "Hosp 4\n(Initial)",
        "FedAvg\n(Initial)", "FedProx\n(Initial)", "FedAvg+DP\n(Initial)",
        "Traditional\n80:20 (Current)"
    ]
    acc = df_matrix["Accuracy (%)"].values
    f1 = df_matrix["F1-Score"].values * 100
    auc = df_matrix["ROC-AUC"].values * 100
    recall = df_matrix["Recall / Sens (%)"].values

    x = np.arange(len(models))
    w = 0.20

    ax1.bar(x - 1.5*w, acc, w, label="Accuracy (%)", color="#2b5c8f")
    ax1.bar(x - 0.5*w, f1, w, label="F1-Score (%)", color="#2ca02c")
    ax1.bar(x + 0.5*w, auc, w, label="ROC-AUC (%)", color="#d95f02")
    ax1.bar(x + 1.5*w, recall, w, label="Recall / Sens (%)", color="#7570b3")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("Performance (%)", fontsize=11)
    ax1.set_ylim(40, 105)
    ax1.set_title("Initial Results vs. Current Traditional 80:20 Benchmark", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # 2. Table panel
    ax2.axis("off")
    table_data = []
    for idx, r in df_matrix.iterrows():
        table_data.append([
            r["Model / Method"].replace(" (McMahan et al.)", "").replace(" (HeartDiseaseMLP)", ""),
            f"{r['Accuracy (%)']:.1f}%",
            f"{r['F1-Score']:.4f}",
            f"{r['ROC-AUC']:.4f}",
            r["Confusion Matrix [TN,FP,FN,TP]"],
            "No" if "None" in r["Privacy Protection"] else "Yes"
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=["Model / Method", "Accuracy", "F1", "ROC-AUC", "Confusion [TN,FP,FN,TP]", "Private?"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.15, 1.6)

    # Highlight current result row (row 8, 1-indexed in table cells is 8)
    for col_idx in range(6):
        cell = table[(8, col_idx)]
        cell.set_facecolor("#e8f4f8")
        cell.get_text().set_fontweight("bold")

    ax2.set_title("Master Evaluation Matrix Table", fontsize=12, fontweight="bold")

    plt.tight_layout()
    matrix_chart_path = OUT_DIR / "initial_vs_current_matrix_chart.png"
    fig.savefig(matrix_chart_path, dpi=200)
    plt.close(fig)
    print(f"[SAVED] Matrix Visualization Chart: {matrix_chart_path}")

    # 3. Dedicated Confusion Matrix Comparison (Side-by-Side)
    cm_fedavg = np.array([[26, 8], [8, 20]])
    cm_trad = np.array([[26, 7], [3, 25]])

    fig, (ax_fl, ax_tr) = plt.subplots(1, 2, figsize=(11, 4.5))

    classes = ["No HD", "HD"]
    im1 = ax_fl.imshow(cm_fedavg, cmap=plt.cm.Blues, interpolation="nearest")
    ax_fl.set_title("Initial Federated Learning (FedAvg)\n[TN=26, FP=8, FN=8, TP=20]", fontweight="bold", fontsize=11)
    ax_fl.set_xticks([0, 1])
    ax_fl.set_yticks([0, 1])
    ax_fl.set_xticklabels(classes)
    ax_fl.set_yticklabels(classes)
    ax_fl.set_ylabel("Actual Label")
    ax_fl.set_xlabel("Predicted Label")
    for i in range(2):
        for j in range(2):
            ax_fl.text(j, i, str(cm_fedavg[i, j]), ha="center", va="center",
                       color="white" if cm_fedavg[i, j] > 15 else "black", fontweight="bold", fontsize=14)

    im2 = ax_tr.imshow(cm_trad, cmap=plt.cm.Greens, interpolation="nearest")
    ax_tr.set_title("Current Traditional Centralized (80:20)\n[TN=26, FP=7, FN=3, TP=25]", fontweight="bold", fontsize=11)
    ax_tr.set_xticks([0, 1])
    ax_tr.set_yticks([0, 1])
    ax_tr.set_xticklabels(classes)
    ax_tr.set_yticklabels(classes)
    ax_tr.set_ylabel("Actual Label")
    ax_tr.set_xlabel("Predicted Label")
    for i in range(2):
        for j in range(2):
            ax_tr.text(j, i, str(cm_trad[i, j]), ha="center", va="center",
                       color="white" if cm_trad[i, j] > 15 else "black", fontweight="bold", fontsize=14)

    plt.tight_layout()
    cm_comp_path = OUT_DIR / "confusion_matrices_comparison.png"
    fig.savefig(cm_comp_path, dpi=200)
    plt.close(fig)
    print(f"[SAVED] Side-by-Side Confusion Matrices: {cm_comp_path}")


if __name__ == "__main__":
    main()
