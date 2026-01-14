import numpy as np
from quantum_torch.qloss import mse


class QTrainer:
    """
    Backward engine for Quantum-Torch using finite difference gradients.
    """

    def __init__(self, model, eps=1e-4):
        self.model = model
        self.eps = eps

    def backward(self, x, y_true):
        params = self.model.parameters()
        grads = np.zeros_like(params)

        for i in range(len(params)):
            original = params[i]

            # theta + eps
            params[i] = original + self.eps
            loss_plus = mse(self.model.forward(x), y_true)

            # theta - eps
            params[i] = original - self.eps
            loss_minus = mse(self.model.forward(x), y_true)

            # restore
            params[i] = original

            grads[i] = (loss_plus - loss_minus) / (2 * self.eps)

        return grads
