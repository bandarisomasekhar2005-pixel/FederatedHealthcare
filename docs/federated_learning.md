# Federated Learning

The project retains the existing PyTorch MLP and server/client implementation.

- Input: 22 processed features.
- Local epochs: 4.
- Batch size: 16.
- Learning rate: 0.01.
- Rounds: 20.
- FedAvg uses sample-weighted parameter aggregation.
- FedProx uses mu = 0.01 and adds `(mu/2)||w-w_t||^2` to the local objective.

The application can launch background experiments through `POST /api/experiments/start`.
