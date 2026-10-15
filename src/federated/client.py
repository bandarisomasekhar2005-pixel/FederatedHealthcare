"""
client.py
---------
Hospital Client simulation node for Federated Healthcare learning.
Performs local model training with support for:
  - Standard Empirical Risk Minimization (FedAvg)
  - Proximal regularization term (FedProx)
  - Local Differential Privacy (DP-SGD weight perturbation)
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.mlp import HeartDiseaseMLP
from src.utils.metrics import evaluate_clinical_predictions
from src.federated.dp import DifferentialPrivacyEngine


class HospitalClient:
    """
    Represents an independent hospital institution holding private patient data.
    """
    def __init__(self, client_id: str, data_path: Path, input_dim: int = 22):
        self.client_id = client_id
        self.data_path = Path(data_path)
        self.input_dim = input_dim

        # Load local private dataset
        df = pd.read_csv(self.data_path)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.float32).reshape(-1, 1)

        self.num_samples = len(X)
        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.y_tensor = torch.tensor(y, dtype=torch.float32)

        self.model = HeartDiseaseMLP(input_dim=self.input_dim)
        self.criterion = nn.BCELoss()

    def __len__(self):
        return self.num_samples

    def train(self, global_weights: dict, epochs: int = 5, batch_size: int = 16,
              lr: float = 0.01, mu: float = 0.0, dp_engine: DifferentialPrivacyEngine = None) -> tuple:
        """
        Executes local training initialized from global server weights.

        Parameters:
          global_weights : Model state_dict received from the central server.
          epochs : Number of local training epochs.
          batch_size : Mini-batch size.
          lr : Local learning rate.
          mu : Proximal regularization coefficient (mu = 0 for FedAvg, mu > 0 for FedProx).
          dp_engine : Optional DifferentialPrivacyEngine for LDP perturbation.

        Returns:
          (updated_weights, num_samples, average_loss)
        """
        # Synchronize local model with global server weights
        self.model.set_weights(global_weights)
        self.model.train()

        # Cache global tensors for FedProx proximal loss calculation
        global_param_tensors = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        dataset = TensorDataset(self.X_tensor, self.y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        total_loss = 0.0
        total_steps = 0

        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)

                # FedProx Proximal Term: (mu / 2) * ||w - w^t||^2
                if mu > 0.0:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        proximal_term += torch.sum((param - global_param_tensors[name]) ** 2)
                    loss += (mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)

        # Retrieve trained local weights
        updated_weights = self.model.get_weights()

        # Apply Local Differential Privacy if requested
        if dp_engine is not None:
            updated_weights = dp_engine.privatize_update(global_weights, updated_weights)

        return updated_weights, self.num_samples, avg_loss

    def evaluate(self, weights: dict = None, X_eval: torch.Tensor = None, y_eval: torch.Tensor = None) -> dict:
        """Evaluates the model on local data or an external evaluation set."""
        if weights is not None:
            self.model.set_weights(weights)

        if X_eval is None:
            X_eval, y_eval = self.X_tensor, self.y_tensor

        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_eval).numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            y_true = y_eval.numpy().ravel()

        return evaluate_clinical_predictions(y_true, preds, probs)
