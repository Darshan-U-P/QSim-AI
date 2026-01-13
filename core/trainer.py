import numpy as np
from core.loss import mse

class QuantumTrainer:
    def __init__(self, model, lr=0.1, eps=1e-4, output_transform=None):
        """
        output_transform: function that maps quantum output -> class output
        """
        self.model = model
        self.lr = lr
        self.eps = eps
        self.output_transform = output_transform

    def step(self, x, y_true):
        # Forward
        q_pred = self.model.forward(x)

        # Convert quantum output to class output if needed
        if self.output_transform is not None:
            y_pred = self.output_transform(q_pred)
        else:
            y_pred = q_pred

        base_loss = mse(y_pred, y_true)
        grads = np.zeros_like(self.model.weights)

        for i in range(len(self.model.weights)):
            original = self.model.weights[i]

            # theta + eps
            self.model.weights[i] = original + self.eps
            q_plus = self.model.forward(x)
            # Apply transform before calculating loss
            y_plus = self.output_transform(q_plus) if self.output_transform else q_plus
            loss_plus = mse(y_plus, y_true)

            # theta - eps
            self.model.weights[i] = original - self.eps
            q_minus = self.model.forward(x)
            # Apply transform before calculating loss
            y_minus = self.output_transform(q_minus) if self.output_transform else q_minus
            loss_minus = mse(y_minus, y_true)

            # Restore
            self.model.weights[i] = original

            grads[i] = (loss_plus - loss_minus) / (2 * self.eps)

        # Update
        self.model.weights -= self.lr * grads

        return base_loss