import numpy as np
import torch


class HybridBatchCoTrainer:
    """
    Batch co-learning trainer:
    Quantum + PB-ANN + Head all learn together from batches.

    Now also acts as a COGNITIVE ENCODER:
    - encode() → (PB cognition, Quantum perception)
    - embed()  → PB cognition only
    """

    def __init__(self, model, eps=1e-4, q_lr=0.1, pb_lr=1e-3):
        self.model = model
        self.eps = eps
        self.q_lr = q_lr
        self.pb_lr = pb_lr
        self.device = next(model.parameters()).device

        # PB + Head optimizer
        self.pb_optimizer = torch.optim.Adam(
            list(self.model.hybrid.pb.parameters()) +
            list(self.model.head.parameters()),
            lr=self.pb_lr
        )

    # ==========================
    # Training
    # ==========================
    def compute_batch_loss(self, batch, criterion):
        total = 0
        for x, y in batch:
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            y = torch.tensor([y], device=self.device)

            logits = self.model(x)
            loss = criterion(logits.unsqueeze(0), y)
            total += loss
        return total / len(batch)

    def step(self, batch, criterion):
        """
        One batch update for both PB and Quantum.
        """
        # --- PB + Head gradient update (Torch) ---
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

            # θ + ε
            q_weights[i] = original + self.eps
            loss_plus = self.compute_batch_loss(batch, criterion).item()

            # θ - ε
            q_weights[i] = original - self.eps
            loss_minus = self.compute_batch_loss(batch, criterion).item()

            # restore
            q_weights[i] = original
            grads[i] = (loss_plus - loss_minus) / (2 * self.eps)

        # Gradient descent step on quantum parameters
        q_weights[:] = q_weights - self.q_lr * grads
        return base_loss

    # ===============================
    # Cognitive Interface (CORE)
    # ===============================
    def encode(self, x):
        """
        Raw perception:
        Returns:
        - pb_out : torch.Tensor [PB_DIM]
        - q_feat : numpy.ndarray [quantum features]
        """
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            pb_out, q_feat = self.model.hybrid(x)

            # PB is torch → detach safely
            if isinstance(pb_out, torch.Tensor):
                pb_out = pb_out.detach().cpu()

            # Quantum is numpy → leave it as-is; if it's a tensor, convert to numpy
            if isinstance(q_feat, torch.Tensor):
                q_feat = q_feat.detach().cpu().numpy()

            return pb_out, q_feat

    def embed(self, x):
        """
        Returns only the PB-ANN cognitive embedding.
        This is what you store in memory and use for reasoning.
        """
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            pb_out, _ = self.model.hybrid(x)
            if isinstance(pb_out, torch.Tensor):
                pb_out = pb_out.detach().cpu()

            return pb_out
