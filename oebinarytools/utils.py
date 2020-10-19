import numpy as np


def norm(x):
    x = np.array(x).astype("float32")
    return (x - np.min(x)) / (np.max(x) - np.min(x))
