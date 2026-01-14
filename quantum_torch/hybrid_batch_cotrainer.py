import numpy as np
import torch

class HybridBatchCoTrainer:
    """
    Batch co-learning trainer:
    Quantum + PB-ANN + Head all learn together from batches.
    """

    def __init__(self, model, eps=1e-4, q_lr=0.1, pb_lr=1e-3):
        self.model = model
        self.eps = eps
        self.q_lr = q_lr
        self.pb_lr = pb_lr

        self.pb_optimizer = torch.optim.Adam(
            list(self.model.hybrid.pb.parameters()) +
            list(self.model.head.parameters()),
            lr=self.pb_lr
        )

    def compute_batch_loss(self, batch, criterion):
        total = 0
        for x, y in batch:
            x = torch.tensor(x, dtype=torch.float32)
            logits = self.model(x)
            loss = criterion(logits.unsqueeze(0), torch.tensor([y]))
            total += loss
        return total / len(batch)

    def step(self, batch, criterion):
        """
        One batch update for both PB and Quantum.
        """
        # --- PB gradient update (Torch) ---
        self.pb_optimizer.zero_grad()
        batch_loss = self.compute_batch_loss(batch, criterion)
        batch_loss.backward()
        self.pb_optimizer.step()

        # --- Quantum finite-difference update ---
        q_weights = self.model.hybrid.qmodel.layer.weights
        grads = np.zeros_like(q_weights)

        base_loss = batch_loss.item()

        for i in range(len(q_weights)):
            original = q_weights[i]

            q_weights[i] = original + self.eps
            loss_plus = self.compute_batch_loss(batch, criterion).item()

            q_weights[i] = original - self.eps
            loss_minus = self.compute_batch_loss(batch, criterion).item()

            q_weights[i] = original
            grads[i] = (loss_plus - loss_minus) / (2 * self.eps)

        q_weights -= self.q_lr * grads

        return base_loss

    # ===============================
    # Embedding extraction interface
    # ===============================
    def embed(self, x):
        """
        Returns the PB-ANN cognitive embedding for an input.
        Shape: (PB_DIM,)
        """
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            pb_out, _ = self.model.hybrid(x)
            return pb_out
