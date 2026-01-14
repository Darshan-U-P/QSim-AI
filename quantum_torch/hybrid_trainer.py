import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from quantum_torch.hybrid import HybridQuantumPB


class HybridQuantumPBTrainer(nn.Module):
    """
    Hybrid Trainer:
    Quantum-Torch (frozen)
        ↓
    PB-ANN (trainable)
        ↓
    Classifier Head
    """

    def __init__(self, n_qubits, pb_dim, num_classes=2, device="cpu"):
        super().__init__()
        self.device = device

        # Hybrid core
        self.hybrid = HybridQuantumPB(n_qubits, pb_dim, device)

        # Freeze quantum parameters
        for param in self.hybrid.qmodel.layer.weights:
            pass  # Quantum stays fixed for now

        # Classification head
        self.head = nn.Linear(pb_dim, num_classes).to(device)

    def forward(self, x):
        pb_out, q_feat = self.hybrid(x)

        # pb_out shape: [1, 1, pb_dim] → squeeze
        pb_out = pb_out.squeeze(0).squeeze(0)

        logits = self.head(pb_out)
        return logits
