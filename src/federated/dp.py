"""
dp.py
-----
Differential Privacy mechanisms for privacy-preserving federated learning.
Implements the Gaussian Mechanism for local model updates:
  1. L2 Norm Clipping with threshold C
  2. Calibrated Gaussian Noise injection based on privacy budget (epsilon, delta)
  3. Privacy accounting
"""

import math
import copy
import torch


class DifferentialPrivacyEngine:
    """
    Applies Local Differential Privacy (LDP) to client weight updates.
    Ensures that transmitted parameter deltas do not leak individual patient records.
    """
    def __init__(self, clip_norm: float = 1.0, noise_multiplier: float = 0.5, delta: float = 1e-5):
        """
        Parameters:
          clip_norm : Maximum L2 norm threshold C for weight updates.
          noise_multiplier : Ratio of Gaussian standard deviation to clipping norm (sigma = noise_multiplier * C).
          delta : Privacy relaxation parameter delta (usually 1e-5 in healthcare).
        """
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta

    @classmethod
    def from_epsilon(cls, epsilon: float, delta: float = 1e-5, clip_norm: float = 1.0):
        """
        Initializes DP Engine given target epsilon and delta via standard Gaussian mechanism formula:
          sigma = sqrt(2 * ln(1.25 / delta)) / epsilon
        """
        sigma = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        return cls(clip_norm=clip_norm, noise_multiplier=sigma, delta=delta)

    def compute_effective_epsilon(self, num_rounds: int, sample_ratio: float = 1.0) -> float:
        """
        Approximates cumulative epsilon over multiple federated rounds using
        advanced composition theorem:
          eps_total = q * sqrt(2 * T * ln(1 / delta)) * (1 / noise_multiplier)
        """
        if self.noise_multiplier <= 0:
            return float("inf")
        # Standard analytical bound for Gaussian mechanism
        base_eps = math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.noise_multiplier
        # Composition over rounds
        eps_total = base_eps * math.sqrt(num_rounds) * sample_ratio
        return round(float(eps_total), 3)

    def privatize_update(self, global_weights: dict, local_weights: dict) -> dict:
        """
        Computes the weight delta: delta = local_weights - global_weights.
        Clips the delta to L2 norm C, injects Gaussian noise N(0, (sigma * C)^2),
        and returns the privatized local weights.
        """
        if self.noise_multiplier <= 0:
            return copy.deepcopy(local_weights)

        # 1. Compute weight update deltas
        deltas = {}
        total_norm_sq = 0.0
        for name in local_weights:
            delta = (local_weights[name].float() - global_weights[name].float())
            deltas[name] = delta
            total_norm_sq += torch.sum(delta ** 2).item()

        total_norm = math.sqrt(total_norm_sq)

        # 2. Clip update deltas to threshold C
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-7))

        # 3. Add calibrated Gaussian noise and re-construct privatized model
        privatized_weights = {}
        std_dev = self.noise_multiplier * self.clip_norm

        for name in local_weights:
            clipped_delta = deltas[name] * clip_coef
            # In-place Gaussian noise injection
            noise = torch.randn_like(clipped_delta) * std_dev
            noised_delta = clipped_delta + noise
            # Reconstruct model parameters: w_priv = w_global + delta_priv
            privatized_weights[name] = (global_weights[name].float() + noised_delta).to(local_weights[name].dtype)

        return privatized_weights
