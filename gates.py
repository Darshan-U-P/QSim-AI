import numpy as np

# Identity Gate
I = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

# Pauli-X Gate (Quantum NOT)
X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

# Pauli-Z Gate (Phase flip)
Z = np.array([
    [1,  0],
    [0, -1]
], dtype=complex)

# Hadamard Gate (Superposition)
H = (1 / np.sqrt(2)) * np.array([
    [1,  1],
    [1, -1]
], dtype=complex)
