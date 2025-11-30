import numpy as np


def generate_linear_data(m=4.0, b=7.0, n=50, x_min=0, x_max=10, noise=2.0, seed=None):
if seed is not None:
np.random.seed(seed)
X = np.linspace(x_min, x_max, n)
y = m * X + b + np.random.randn(n) * noise
return X, y
