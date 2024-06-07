import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def cross_entropy(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-8,1-1e-8)
    return np.mean(-y_true * np.log(y_pred_clipped))

def cross_entropy_prime(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-8,1-1e-8)
    return (-y_true/ y_pred_clipped) / np.size(y_true)



