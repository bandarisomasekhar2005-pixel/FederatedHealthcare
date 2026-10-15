# Privacy-Preserving Federated Learning for Healthcare

A comprehensive final-year B.Tech Computer Science and Engineering project implementing a privacy-preserving federated learning framework tailored for distributed healthcare institutions.

---

## 1. Project Overview & Architecture

Healthcare institutions are legally restricted by data privacy regulations (HIPAA, GDPR) from centralizing patient electronic health records. This framework simulates four distinct hospital nodes collaborating to train a shared cardiovascular disease diagnostic model without exposing private clinical observations.

### Complete Architecture Tree
```text
FederatedHealthcare/
│
├── data/
│   ├── raw/                                  # Pristine raw UCI dataset
│   │   ├── processed.cleveland.data          # Original Cleveland dataset from UCI
│   │   ├── heart-disease.names               # UCI attribute documentation & codebook
│   │   ├── heart_disease_raw.zip             # Downloaded official zip archive
│   │   └── cleveland_raw.csv                 # Raw CSV with standard column headers
│   │
│   ├── processed/                            # Preprocessed & standardized data
│   │   └── heart_disease_processed.csv       # 303 samples, 22 features + 1 binary target
│   │
│   ├── clients/                              # Partitioned hospital nodes (Non-IID)
│   │   ├── hospital_1.csv                    # Hospital 1 local dataset (108 samples)
│   │   ├── hospital_2.csv                    # Hospital 2 local dataset (51 samples)
│   │   ├── hospital_3.csv                    # Hospital 3 local dataset (49 samples)
│   │   └── hospital_4.csv                    # Hospital 4 local dataset (95 samples)
│   │
│   └── splits/                               # 80/20 train/test local partitions
│       ├── hospital_1_train.csv ... hospital_4_train.csv
│       ├── global_train.csv                  # Pooled training data (242 samples)
│       └── global_test.csv                   # Pooled holdout test data (61 samples)
│
├── src/                                      # Modular Python Package
│   ├── models/
│   │   └── mlp.py                            # Clinical Multi-Layer Perceptron (LayerNorm + Dropout)
│   ├── federated/
│   │   ├── client.py                         # HospitalClient (local SGD + FedProx + DP)
│   │   ├── server.py                         # FederatedServer (FedAvg / FedProx aggregation)
│   │   └── dp.py                             # Differential Privacy engine (Gaussian mechanism)
│   └── utils/
│       └── metrics.py                        # Clinical diagnostics (Sens, Spec, Prec, F1, AUC)
│
├── scripts/
│   ├── download_dataset.py                   # Downloads real UCI Cleveland dataset
│   ├── preprocess_dataset.py                 # Imputation, encoding, scaling, binarization
│   ├── create_clients.py                     # Dirichlet Non-IID hospital partitioning
│   ├── verify_dataset.py                     # Multi-stage automated validation & reporting
│   ├── run_pipeline.py                       # Single-command end-to-end dataset pipeline
│   └── train_federated.py                    # Complete Phase 2 federated benchmark
│
├── results/
│   ├── dataset/                              # Dataset verification reports & plots
│   │   ├── client_distribution_plot.png
│   │   ├── client_distribution_summary.csv
│   │   └── verification_report.txt
│   │
│   └── federated/                            # Experimental benchmark results
│       ├── comparative_metrics.csv           # Performance metrics across all 9 paradigms
│       ├── federated_convergence.png         # Round-by-round convergence curves
│       ├── model_comparison_bar.png          # Comparative performance bar chart
│       └── federated_experiment_report.txt   # Detailed experimental analysis report
│
└── README.md
```

---

## 2. Dataset & Clinical Preprocessing

* **Dataset Source**: [UCI Machine Learning Repository — Heart Disease (Cleveland Subset)](https://archive.ics.uci.edu/dataset/45/heart+disease)
* **Authenticity**: 100% Real clinical observations (303 patients). Zero fabricated or synthetic data.
* **Preprocessed Feature Dimensions**: 22 standardized inputs + 1 binary target (`0`: No Heart Disease, `1`: Heart Disease Present).
* **Missing Value Imputation**: Median for `ca` (fluoroscopy vessels, 4 missing), Mode for `thal` (thallium defect, 2 missing).
* **Feature Scaling**: `LayerNorm`-compatible standardization on continuous variables (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`). One-hot encoding on categorical nominal features (`cp`, `restecg`, `slope`, `thal`).

---

## 3. Four Non-IID Hospital Clients (Dirichlet $\alpha = 0.5$)

To simulate real-world data heterogeneity across healthcare clinics, we partition the dataset using a **Dirichlet distribution** ($\mathbf{p}_c \sim \text{Dirichlet}(\alpha=0.5)$):

| Hospital Node | Demographics Profile | Sample Count | Class 0 (No HD) | Class 1 (HD Present) | HD Prevalence |
|---|---|:---:|:---:|:---:|:---:|
| **Hospital 1** | Regional General Hospital | 108 (35.6%) | 62 (57.4%) | 46 (42.6%) | 42.6% |
| **Hospital 2** | Specialized Critical Cardiac Center | 51 (16.8%) | 12 (23.5%) | 39 (76.5%) | 76.5% |
| **Hospital 3** | Inpatient Cardiology Unit | 49 (16.2%) | 20 (40.8%) | 29 (59.2%) | 59.2% |
| **Hospital 4** | Outpatient Diagnostic & Wellness Clinic | 95 (31.4%) | 70 (73.7%) | 25 (26.3%) | 26.3% |
| **Global** | **Pooled Dataset** | **303 (100%)** | **164 (54.1%)** | **139 (45.9%)** | **45.9%** |

* Chi-Square Test of Independence: $\chi^2 = 37.8282, p = 3.0734 \times 10^{-8}$ ($p \ll 0.01$), confirming statistically significant non-IID data distribution.

---

## 4. Federated Learning & Privacy Formulations

### A. FedAvg (Federated Averaging — McMahan et al., 2017)
Each hospital $k$ performs $E$ local epochs of gradient descent on its local loss $F_k(w)$. The central server aggregates updates via sample-weighted averaging:
$$w^{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_k^{t+1}$$

### B. FedProx (Proximal Regularization — Li et al., 2020)
To combat **client drift** caused by non-IID label skew, FedProx introduces a proximal regularization term to each client's local loss function:
$$\min_w h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \| w - w^t \|_2^2$$
where $\mu = 0.01$. The proximal term penalizes local weights that stray too far from the global consensus $w^t$.

### C. Local Differential Privacy (DP-SGD)
Before transmitting weight deltas $\Delta w_k = w_k - w^t$ to the central server, the local client applies:
1. **L2 Norm Clipping**:
   $$\tilde{\Delta}_k = \Delta_k \cdot \min\left(1, \frac{C}{\|\Delta_k\|_2}\right)$$
2. **Calibrated Gaussian Noise Addition**:
   $$\Delta_k^{\text{priv}} = \tilde{\Delta}_k + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$$
This prevents gradient inversion, membership inference, and model inversion attacks.

---

## 5. Final Federated Learning Benchmark Results

The final preserved benchmark is the **stratified 80:20 global holdout experiment** documented in
`results/federated/holdout_80_20/final_test_metrics.csv` and
`results/federated/federated_experiment_report.txt`. It uses 242 training samples and 61 held-out
test samples from the 303-record UCI Cleveland heart-disease dataset. Preprocessing parameters were
fitted on training data only, and the held-out test set was not used during federated training.

| Algorithm | Accuracy | Precision | Recall / Sensitivity | Specificity | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| FedAvg | 83.61% | 78.12% | 89.29% | 78.79% | 83.33% | 90.80% |
| FedProx ($\\mu = 0.01$) | 83.61% | 80.00% | 85.71% | 81.82% | 82.76% | 93.51% |
| FedAvg + DP | 83.61% | 78.12% | 89.29% | 78.79% | 83.33% | 93.94% |
| **FedProx + DP** ($\\mu = 0.01$) | **86.89%** | **83.33%** | **89.29%** | **84.85%** | **86.21%** | **94.26%** |

### Interpretation

- **Best accuracy in this benchmark:** FedProx + DP, at 86.89%.
- **Best ROC-AUC in this benchmark:** FedProx + DP, at 94.26%.
- Recall/sensitivity for FedProx + DP is 89.29%; specificity is 84.85%.
- These are results from a small holdout set (61 cases). They should be interpreted as research-prototype results, not evidence of clinical readiness.

### Note on earlier results

An earlier exploratory table in previous project documentation reported a different set of values and
included isolated-hospital and centralized baselines. Those values are **not mixed with this final
80:20 holdout benchmark**. The final benchmark table above is the preserved result set used by
`results/latest_results.json`, `results/latest_results.csv`, and the completion report. To report
the earlier experiment as a separate result, its exact dataset split, configuration, and provenance
should be documented separately.

## 6. How to Run

### Run Phase 1: Dataset Pipeline
```powershell
python scripts/run_pipeline.py
```

### Run Phase 2: Federated Learning & Privacy Experiments
```powershell
python scripts/train_federated.py
```
Outputs are automatically written to `results/federated/`:
- `comparative_metrics.csv`
- `federated_convergence.png`
- `model_comparison_bar.png`
- `federated_experiment_report.txt`


# End-to-End Application (October 2026)

The project now includes a React/Vite research dashboard, FastAPI backend, SQLite database,
actual model inference, SHAP explanations, experiment history, audit logging, a secure
aggregation prototype, and a small homomorphic-encryption demonstration.

## Run on Windows

```bat
setup.bat
start_all.bat
```

URLs:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

## Transparency labels

- **Implemented:** UCI dataset pipeline, non-IID hospital simulation, PyTorch MLP, FedAvg,
  FedProx, update-level DP, held-out evaluation, model persistence, FastAPI, SQLite, React/Vite,
  actual prediction and SHAP.
- **Prototype:** Secure aggregation mask-cancellation demonstration.
- **Demonstration:** Small additive Paillier homomorphic-encryption demonstration.
- **Future work:** formal per-example DP accounting, production-grade secure aggregation,
  external clinical validation, and full HE-encrypted neural-network training.

## Reported experiment

`results/latest_results.json` and `results/latest_results.csv` preserve the final verified
80:20 holdout results. They are intentionally not overwritten by interactive reruns. New
interactive experiments are stored separately in SQLite and marked as new executions.

## Medical disclaimer

This application is a research and educational prototype. It is not a clinically validated
diagnostic system and must not be used as a substitute for professional medical advice.
