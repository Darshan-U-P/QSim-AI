import numpy as np
from core.loss import mse


class QuantumTrainer:
    def __init__(self, model, lr=0.1, eps=1e-4, output_transform=None):
        """
        model            : QuantumLayer instance
        lr               : learning rate
        eps              : finite difference step
        output_transform : function that maps quantum output -> class output
                           (for example 4 quantum states -> 2 classes)
        """
        self.model = model
        self.lr = lr
        self.eps = eps
        self.output_transform = output_transform

    def _forward_loss(self, x, y_true):
        """
        Forward pass + loss calculation with optional output transform
        """
        q_pred = self.model.forward(x)
        if self.output_transform:
            y_pred = self.output_transform(q_pred)
        else:
            y_pred = q_pred
        return mse(y_pred, y_true)

    def _compute_gradients(self, x, y_true):
        """
        Compute gradients for a single sample using finite difference
        """
        grads = np.zeros_like(self.model.weights)

        for i in range(len(self.model.weights)):
            original = self.model.weights[i]

            # f(theta + eps)
            self.model.weights[i] = original + self.eps
            loss_plus = self._forward_loss(x, y_true)

            # f(theta - eps)
            self.model.weights[i] = original - self.eps
            loss_minus = self._forward_loss(x, y_true)

            # Restore original weight
            self.model.weights[i] = original

            # Central difference
            grads[i] = (loss_plus - loss_minus) / (2 * self.eps)

        return grads

    def step(self, x, y_true):
        """
        Single-sample training step (already working before)
        """
        base_loss = self._forward_loss(x, y_true)
        grads = self._compute_gradients(x, y_true)

        # Gradient descent update
        self.model.weights -= self.lr * grads
        return base_loss

    def train_batch(self, dataset):
        """
        Batch training:
        dataset = [(x1, y1), (x2, y2), ...]

        We:
        - Compute gradients for each sample
        - Average them
        - Update weights once per batch
        """
        total_grads = np.zeros_like(self.model.weights)
        total_loss = 0.0

        for x, y_true in dataset:
            loss = self._forward_loss(x, y_true)
            grads = self._compute_gradients(x, y_true)

            total_grads += grads
            total_loss += loss

        # Average gradients and loss
        total_grads /= len(dataset)
        total_loss /= len(dataset)

        # Update once per batch
        self.model.weights -= self.lr * total_grads

        return total_loss
