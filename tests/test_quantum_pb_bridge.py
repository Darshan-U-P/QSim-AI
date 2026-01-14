import numpy as np
import torch
from core.quantum_pb_bridge import QuantumToPB

# Configuration
NUM_QUBITS = 2
PB_OUT_DIM = 64

# Create hybrid model
model = QuantumToPB(
    num_qubits=NUM_QUBITS,
    pb_out_dim=PB_OUT_DIM,
    device="cpu"
)

# Sample classical input
x = np.array([0.7, 1.2])

# Forward pass
pb_output = model(x)

print("Input:", x)
print("Quantum features:", model.quantum_features(x))
print("PB output shape:", pb_output.shape)
print("PB output (first 10 neurons):", pb_output[0][:10])
