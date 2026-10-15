import numpy as np
from backend.app.security.secure_aggregation import demo as secure_demo
from backend.app.security.he_demo import demo as he_demo

def test_secure_aggregation():
    r=secure_demo()
    assert r["masks_cancel"]
    assert np.allclose(r["plain_sum"],r["masked_sum"])

def test_he_demo():
    r=he_demo((3,7,11))
    assert r["homomorphic_addition_verified"]
    assert r["decrypted_sum"]==21
