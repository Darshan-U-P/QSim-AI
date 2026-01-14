import numpy as np
from core.quantum_layer import QuantumLayer

# Quantum feature extractor (2 qubits → 4 features)
quantum = QuantumLayer(2)

# Simple classical linear classifier
# weights: (4 → 2 classes)
W = np.random.randn(4, 2)
b = np.zeros(2)

def classical_forward(features):
    logits = features @ W + b
    # softmax
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()

# Sample input
x = np.array([0.7, 1.2])

# Step 1: Quantum features
features = quantum.forward(x)

# Step 2: Classical decision
y = classical_forward(features)

print("Quantum features:", features)
print("Classical output:", y)
print("Sum of quantum features:", features.sum())
print("Sum of class probs:", y.sum())
