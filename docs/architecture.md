# Architecture

Frontend (React + Vite) calls the FastAPI REST API. FastAPI persists experiment metadata,
metrics, predictions and audit logs in SQLite and calls the existing PyTorch federated-learning
modules for training/inference.

Flow:

Patients -> simulated hospital clients -> local training -> FedAvg/FedProx -> update-level DP
-> secure aggregation prototype -> global model -> held-out evaluation -> prediction + SHAP -> dashboard.

The 20% test set is never passed to the federated training rounds.
