# app/utils.py
import numpy as np

def generate_linear_data(m=4.0, b=7.0, n=50, noise=2.0, seed=None, x_min=0.0, x_max=10.0):
    """
    Generate synthetic linear data: y = m*x + b + N(0, noise)
    seed: if 0 -> deterministic constant seed; if None -> random
    """
    if seed is None:
        seed = np.random.randint(0, 10_000_000)
    if seed == 0:
        seed = 42
    rng = np.random.RandomState(seed)
    X = rng.uniform(x_min, x_max, size=n)
    X = np.sort(X)
    y = m * X + b + rng.randn(n) * noise
    return X, y
