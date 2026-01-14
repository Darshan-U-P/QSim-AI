import numpy as np
import torch
from quantum_torch.hybrid import HybridQuantumPB

# Configuration
NUM_QUBITS = 2
PB_OUT_DIM = 32

model = HybridQuantumPB(
    n_qubits=NUM_QUBITS,
    pb_out_dim=PB_OUT_DIM,
    device="cpu"
)

# Sample inputs
inputs = [
    np.array([0.2, 0.3]),
    np.array([1.5, 0.8]),
    np.array([2.2, 2.4])
]

for x in inputs:
    pb_out, q_feat = model(x)

    print("\nInput:", x)
    print("Quantum features:", q_feat)
    print("Sum of quantum probs:", q_feat.sum())
    print("PB output shape:", pb_out.shape)
    print("PB output (first 10 neurons):", pb_out[0][:10])
