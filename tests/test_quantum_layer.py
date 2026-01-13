import numpy as np
from core.quantum_layer import QuantumLayer

ql = QuantumLayer(2)

# Input data (classical)
x = np.array([0.3, 1.2])

output = ql.forward(x)

print("Quantum Layer Output:")
print(output)
print("Sum of probabilities:", output.sum())
