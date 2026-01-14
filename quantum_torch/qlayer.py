import numpy as np
from quantum_torch.qmodule import QModule
from core.quantum_layer import QuantumLayer


class QuantumLayerModule(QModule):
    """
    Wraps your existing QuantumLayer to behave like a QModule.
    """

    def __init__(self, n_qubits):
        self.layer = QuantumLayer(n_qubits)

    def forward(self, x):
        return self.layer.forward(x)

    def parameters(self):
        return self.layer.weights
