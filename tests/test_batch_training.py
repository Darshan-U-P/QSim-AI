import numpy as np
from core.quantum_layer import QuantumLayer
from core.trainer import QuantumTrainer

# 1-qubit batch training → 2 classes
model = QuantumLayer(1)
trainer = QuantumTrainer(model, lr=0.3)

# Dataset:
# Small angles → class 0
# Large angles → class 1
dataset = [
    (np.array([0.1]), np.array([1.0, 0.0])),
    (np.array([0.2]), np.array([1.0, 0.0])),
    (np.array([2.5]), np.array([0.0, 1.0])),
    (np.array([2.8]), np.array([0.0, 1.0])),
]

print("Initial outputs:")
for x, y in dataset:
    print(x, "→", model.forward(x))

for epoch in range(50):
    loss = trainer.train_batch(dataset)
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Batch Loss: {loss:.6f}")

print("\nFinal outputs:")
for x, y in dataset:
    print(x, "→", model.forward(x))
