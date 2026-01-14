import numpy as np
import torch
import torch.optim as optim


class HybridCoTrainer:
    """
    Joint training:
    PB-ANN + Quantum-Torch learn together.
    """

    def __init__(self, model, quantum_eps=1e-3, quantum_lr=0.2, pb_lr=1e-3):
        self.model = model
        self.q_eps = quantum_eps
        self.q_lr = quantum_lr

        # PB optimizer (normal gradient)
        self.pb_optimizer = optim.Adam(
            list(self.model.hybrid.pb.parameters()) +
            list(self.model.head.parameters()),
            lr=pb_lr
        )

    def compute_loss(self, x, y, criterion):
        logits = self.model(x)
        loss = criterion(logits.unsqueeze(0), torch.tensor([y]))
        return loss

    def quantum_step(self, x, y, criterion):
        q_weights = self.model.hybrid.qmodel.layer.weights
        grads = np.zeros_like(q_weights)

        base_loss = self.compute_loss(x, y, criterion).item()

        for i in range(len(q_weights)):
            original = q_weights[i]

            q_weights[i] = original + self.q_eps
            loss_plus = self.compute_loss(x, y, criterion).item()

            q_weights[i] = original - self.q_eps
            loss_minus = self.compute_loss(x, y, criterion).item()

            q_weights[i] = original
            grads[i] = (loss_plus - loss_minus) / (2 * self.q_eps)

        q_weights -= self.q_lr * grads
        return base_loss

    def pb_step(self, x, y, criterion):
        self.pb_optimizer.zero_grad()
        loss = self.compute_loss(x, y, criterion)
        loss.backward()
        self.pb_optimizer.step()
        return loss.item()
