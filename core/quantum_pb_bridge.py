import torch
import torch.nn as nn
import numpy as np
from core.quantum_layer import QuantumLayer
from moe_transformer2 import PBLinear


class QuantumToPB(nn.Module):
    """
    Bridge between QSim-AI QuantumLayer and PB-ANN (PBLinear).

    Flow:
        Classical Input
            ↓
        QuantumLayer (numpy probabilities)
            ↓
        Torch Tensor Conversion
            ↓
        PBLinear (biological inhibition + sparse firing)
            ↓
        PB-ANN Output
    """

    def __init__(self, num_qubits, pb_out_dim, device="cpu"):
        super().__init__()
        self.num_qubits = num_qubits
        self.quantum = QuantumLayer(num_qubits)
        self.device = device

        # PB-ANN neuron layer
        self.pb = PBLinear(
            in_features=2 ** num_qubits,
            out_features=pb_out_dim
        ).to(device)

    def forward(self, x):
        """
        x: numpy array of shape (num_qubits,)
        returns: torch tensor from PB-ANN
        """

        # 1. Quantum forward (returns numpy probabilities)
        q_features = self.quantum.forward(x)

        # 2. Convert to torch tensor
        q_tensor = torch.tensor(
            q_features,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)   # shape: (1, 2^num_qubits)

        # 3. PB-ANN processing
        pb_output = self.pb(q_tensor)

        return pb_output

    def quantum_features(self, x):
        """
        Return raw quantum probability features (no PB)
        """
        return self.quantum.forward(x)

    def pb_forward_from_quantum(self, q_features):
        """
        Feed precomputed quantum features directly into PB-ANN.
        """
        q_tensor = torch.tensor(
            q_features,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        return self.pb(q_tensor)
