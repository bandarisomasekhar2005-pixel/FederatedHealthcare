"""
server.py
---------
Federated Aggregation Server coordinating collaborative learning across hospitals.
Performs:
  - Global model broadcasting
  - Federated parameter aggregation (FedAvg / FedProx)
  - Client training loss tracking (WITHOUT accessing the unseen test set during training)
  - Final single evaluation on the held-out 20% test set after all rounds conclude
"""

import copy
import torch
import numpy as np
from typing import List, Dict
from src.models.mlp import HeartDiseaseMLP
from src.utils.metrics import evaluate_clinical_predictions
from src.federated.client import HospitalClient
from src.federated.dp import DifferentialPrivacyEngine


class FederatedServer:
    """
    Central server coordinating federated learning rounds without accessing raw patient data.
    """
    def __init__(self, input_dim: int = 22):
        self.input_dim = input_dim
        self.global_model = HeartDiseaseMLP(input_dim=input_dim)
        self.history = []

    def aggregate(self, client_updates: List[tuple]) -> dict:
        """
        Performs Federated Averaging (FedAvg):
          w_{global} = sum_k (n_k / N) * w_k
        where n_k is client sample count and N is total sample count.
        """
        total_samples = sum(num_samples for _, num_samples, _ in client_updates)
        aggregated_weights = copy.deepcopy(client_updates[0][0])

        # Initialize accumulator to zeros
        for key in aggregated_weights:
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key], dtype=torch.float32)

        # Weighted sum across all participating hospitals
        for weights, num_samples, _ in client_updates:
            weight_factor = num_samples / total_samples
            for key in weights:
                aggregated_weights[key] += weights[key].float() * weight_factor

        # Cast back to original parameter types
        base_weights = client_updates[0][0]
        for key in aggregated_weights:
            aggregated_weights[key] = aggregated_weights[key].to(base_weights[key].dtype)

        return aggregated_weights

    def run_rounds(self, clients: List[HospitalClient], num_rounds: int = 20,
                   local_epochs: int = 4, batch_size: int = 16, lr: float = 0.01,
                   mu: float = 0.0, dp_engine: DifferentialPrivacyEngine = None,
                   verbose: bool = True) -> dict:
        """
        Executes T federated communication rounds using ONLY client training data.
        NOTE: The unseen test set is NEVER passed or evaluated during communication rounds.

        Parameters:
          clients : List of HospitalClient instances (holding training data only).
          num_rounds : Number of communication rounds.
          local_epochs : Epochs of local SGD at each hospital.
          batch_size : Local mini-batch size.
          lr : Local learning rate.
          mu : FedProx proximal regularizer (mu = 0 for FedAvg, mu > 0 for FedProx).
          dp_engine : Optional DP engine.
        """
        self.history = []
        algo_name = "FedProx (mu=" + str(mu) + ")" if mu > 0 else "FedAvg"
        if dp_engine is not None:
            eff_eps = dp_engine.compute_effective_epsilon(num_rounds)
            algo_name += f" + DP (eps~{eff_eps})"

        if verbose:
            print(f"\n--- Initiating {algo_name} ({num_rounds} rounds, {len(clients)} hospitals) ---")
            print("    [LEAKAGE-FREE] Unseen test set is NOT evaluated during training rounds.")

        for round_idx in range(1, num_rounds + 1):
            global_weights = self.global_model.get_weights()
            client_updates = []
            client_losses = []

            # 1. Distribute global model to each hospital and perform local training
            for client in clients:
                updated_weights, n_samples, loss = client.train(
                    global_weights=global_weights,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    mu=mu,
                    dp_engine=dp_engine
                )
                client_updates.append((updated_weights, n_samples, loss))
                client_losses.append(loss)

            # 2. Server performs federated aggregation
            new_global_weights = self.aggregate(client_updates)
            self.global_model.set_weights(new_global_weights)

            # 3. Record training convergence metrics (WITHOUT test set evaluation)
            avg_loss = float(np.mean(client_losses))
            round_metrics = {
                "round": round_idx,
                "avg_client_loss": avg_loss
            }
            self.history.append(round_metrics)

            if verbose and (round_idx % 5 == 0 or round_idx == 1 or round_idx == num_rounds):
                print(f"  [Round {round_idx:2d}/{num_rounds:2d}] Average Client Training Loss: {avg_loss:.4f}")

        return {
            "algorithm": algo_name,
            "final_weights": self.global_model.get_weights(),
            "history": self.history
        }

    def evaluate_final_model(self, test_X: np.ndarray, test_y: np.ndarray) -> tuple:
        """
        Evaluates the final global federated model ONCE on the unseen holdout test set.
        Returns: (metrics_dict, predicted_probabilities, binary_predictions)
        """
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
        test_y_arr = np.asarray(test_y).ravel().astype(int)

        self.global_model.eval()
        with torch.no_grad():
            probs = self.global_model(test_X_tensor).numpy().ravel()
            preds = (probs >= 0.5).astype(int)

        metrics = evaluate_clinical_predictions(test_y_arr, preds, probs)
        return metrics, probs, preds
