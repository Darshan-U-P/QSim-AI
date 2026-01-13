import numpy as np
from core.circuit import QuantumCircuit


class QuantumLayer:
    def __init__(self, n_qubits):
        self.n = n_qubits
        # Initialize trainable parameters
        self.weights = np.random.uniform(0, 2*np.pi, size=n_qubits)

    def forward(self, x):
        """
        x: input vector of size n_qubits (classical data)
        returns: probability vector of size 2^n_qubits
        """
        qc = QuantumCircuit(self.n)

        # Encode classical input into quantum state
        for i in range(self.n):
            qc.ry(i, x[i])

        # Apply trainable quantum layer
        for i in range(self.n):
            qc.ry(i, self.weights[i])

        # Output probabilities
        return qc.probabilities()
