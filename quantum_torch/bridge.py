import torch
import torch.nn as nn
import numpy as np
from quantum_torch.qlayer import QuantumLayerModule
from moe_transformer2 import PBLinear


class QuantumTorchToPB(nn.Module):
    """
    Bridge between Quantum-Torch and PB-ANN (PyTorch).
    """

    def __init__(self, n_qubits, pb_out_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.qmodel = QuantumLayerModule(n_qubits)

        self.pb = PBLinear(
            in_features=2 ** n_qubits,
            out_features=pb_out_dim
        ).to(device)

    def forward(self, x):
        """
        x: numpy array
        """
        # Quantum-Torch forward
        q_features = self.qmodel.forward(x)

        # Convert to torch tensor
        q_tensor = torch.tensor(
            q_features,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        # PB-ANN forward
        pb_out = self.pb(q_tensor)
        return pb_out
