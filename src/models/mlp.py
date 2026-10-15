"""
mlp.py
------
Neural network architecture for clinical heart disease risk prediction.
Standard tabular Multi-Layer Perceptron (MLP) compatible with PyTorch.
Utilizes LayerNorm for robust federated training across non-IID clients.
"""

import copy
import torch
import torch.nn as nn


class HeartDiseaseMLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification on processed clinical features.
    Architecture:
      Input (22 features) -> Dense(32) -> LayerNorm(32) -> ReLU -> Dropout(0.2)
                          -> Dense(16) -> LayerNorm(16) -> ReLU -> Dropout(0.1)
                          -> Dense(1)  -> Sigmoid
    """
    def __init__(self, input_dim: int = 22, hidden1: int = 32, hidden2: int = 16, dropout: float = 0.2):
        super(HeartDiseaseMLP, self).__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning predicted probability in range (0, 1)."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation mode forward pass."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary predictions {0, 1}."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()

    def get_weights(self) -> dict:
        """Returns deep copy of the model state dict."""
        return copy.deepcopy(self.state_dict())

    def set_weights(self, weights: dict):
        """Loads state dict into model."""
        self.load_state_dict(copy.deepcopy(weights))

    def clone(self):
        """Returns an exact duplicate instance of this model."""
        cloned = HeartDiseaseMLP(input_dim=self.input_dim)
        cloned.load_state_dict(self.state_dict())
        return cloned
