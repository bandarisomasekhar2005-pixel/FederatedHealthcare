# Privacy and Security

## Update-Level Differential Privacy

The existing implementation clips each model update to L2 bound C=1.0 and adds Gaussian
noise with multiplier 0.25. Delta is 1e-5 and the current 20-round accounting estimate is
epsilon=86.667.

This is explicitly an **update-level privacy accounting estimate**. It must not be described
as a full per-example DP-SGD guarantee.

## Secure Aggregation Prototype

The application contains a mask-cancellation demonstration using cryptographically secure
randomness. It is labeled a prototype and does not claim production-grade protection against
malicious clients, dropouts, or collusion.

## Homomorphic Encryption Demonstration

A small additive Paillier demonstration verifies encrypted addition followed by decryption.
It is not used for the complete neural-network training pipeline.
