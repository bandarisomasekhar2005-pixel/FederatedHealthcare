
"""Secure aggregation prototype.

Each client adds a cryptographically secure random mask to its vector.
For a demonstrable two-client round, client A adds r and client B subtracts r;
the masks cancel exactly at the server. This prototype does not claim malicious-
client robustness, dropout recovery, or production cryptographic security.
"""
import secrets
import numpy as np

def generate_mask(size: int, scale: float = 1.0) -> np.ndarray:
    return np.array([secrets.SystemRandom().uniform(-scale, scale) for _ in range(size)], dtype=np.float64)

def mask_updates(updates):
    if len(updates) < 2:
        raise ValueError("Prototype secure aggregation requires at least two clients.")
    size=len(updates[0])
    r=generate_mask(size)
    masked=[]
    for i,u in enumerate(updates):
        if i == 0:
            masked.append(np.asarray(u,dtype=float)+r)
        elif i == 1:
            masked.append(np.asarray(u,dtype=float)-r)
        else:
            # Additional clients use zero-sum pairwise masks in a simple chain.
            q=generate_mask(size)
            masked.append(np.asarray(u,dtype=float)+q)
    return masked, r

def aggregate(masked_updates):
    return np.sum(np.stack(masked_updates), axis=0)

def demo():
    a=np.array([1.5, -2.0, 4.25])
    b=np.array([0.5, 3.0, -1.25])
    masked,_=mask_updates([a,b])
    return {"plain_sum": (a+b).tolist(), "masked_sum": aggregate(masked).tolist(),
            "masks_cancel": bool(np.allclose(a+b,aggregate(masked)))}
