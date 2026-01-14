import torch
import torch.nn as nn
import numpy as np

from quantum_torch.qlayer import QuantumLayerModule
from moe_transformer2 import PBLinear


class HybridQuantumPB(nn.Module):
    """
    Hybrid Model:
        Quantum-Torch  →  PB-ANN (PBLinear)

    Quantum:
        - Generates probabilistic features (2^n size)

    PB-ANN:
        - Applies inhibition, sparsity, biological neuron behavior
    """

    def __init__(self, n_qubits, pb_out_dim, device="cpu"):
        super().__init__()
        self.device = device

        # Quantum-Torch layer
        self.qmodel = QuantumLayerModule(n_qubits)

        # PB-ANN layer
        self.pb = PBLinear(
            in_features=2 ** n_qubits,
            out_features=pb_out_dim
        ).to(device)

    def forward(self, x):
        """
        x: numpy array (classical input)
        returns: torch tensor (PB-ANN output)
        """

        # 1. Quantum forward (NumPy)
        q_features = self.qmodel.forward(x)

        # 2. Convert to Torch tensor
        q_tensor = torch.tensor(
            q_features,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        # 3. PB-ANN forward
        pb_output = self.pb(q_tensor)

        return pb_output, q_features
