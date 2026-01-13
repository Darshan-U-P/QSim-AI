import numpy as np

# -------------------------
# Fixed Single-Qubit Gates
# -------------------------

# Identity
I = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

# Pauli-X (NOT gate)
X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

# Pauli-Z (Phase flip)
Z = np.array([
    [1,  0],
    [0, -1]
], dtype=complex)

# Hadamard (Creates superposition)
H = (1 / np.sqrt(2)) * np.array([
    [1,  1],
    [1, -1]
], dtype=complex)

# -------------------------
# Parameterized Rotation Gates
# These are trainable gates
# -------------------------

def RX(theta):
    """
    Rotation around X-axis
    """
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)


def RY(theta):
    """
    Rotation around Y-axis
    """
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ], dtype=complex)


def RZ(theta):
    """
    Rotation around Z-axis
    """
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)
