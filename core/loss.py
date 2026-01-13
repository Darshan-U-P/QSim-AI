import numpy as np

def mse(y_pred, y_true):
    """
    Mean Squared Error loss
    """
    return np.mean((y_pred - y_true) ** 2)
