import numpy as np
import torch


class HybridQuantumTrainer:
    """
    Trains only the Quantum-Torch parameters using loss coming from:

        Quantum → PB-ANN → Classifier Head → Loss

    PB-ANN and Classifier Head are completely frozen.
    Quantum-Torch parameters evolve to make cognition easier.

    This implements:
        Loss → Brain (frozen evaluator) → Quantum perception learning
    """

    def __init__(self, model, eps=1e-4, lr=0.1):
        """
        model : HybridQuantumPBTrainer
        eps   : finite difference step size
        lr    : quantum learning rate
        """
        self.model = model
        self.eps = eps
        self.lr = lr


    def compute_loss(self, x, target, criterion):
        """
        Forward through full hybrid system and compute scalar loss.
        """
        logits = self.model(x)
        loss = criterion(logits.unsqueeze(0), torch.tensor([target]))
        return loss.item()

    def step(self, x, target, criterion):
        """
        One optimization step for quantum parameters only.
        Uses finite-difference gradient estimation.
        """
        # Quantum parameters
        q_weights = self.model.hybrid.qmodel.layer.weights

        # Gradient container
        grads = np.zeros_like(q_weights)

        # Base loss
        base_loss = self.compute_loss(x, target, criterion)

        # Finite difference gradient
        for i in range(len(q_weights)):
            original = q_weights[i]

            # θ + ε
            q_weights[i] = original + self.eps
            loss_plus = self.compute_loss(x, target, criterion)

            # θ - ε
            q_weights[i] = original - self.eps
            loss_minus = self.compute_loss(x, target, criterion)

            # Restore original weight
            q_weights[i] = original

            # Central difference
            grads[i] = (loss_plus - loss_minus) / (2 * self.eps)

        # Gradient descent update
        q_weights -= self.lr * grads

        return base_loss
