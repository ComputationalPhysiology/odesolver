import numpy as np


def vander_pool(t, y):
    mu = 1.0
    return np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])


def test_vander_pool():
    ...
