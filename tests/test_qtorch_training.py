import numpy as np
from quantum_torch.qlayer import QuantumLayerModule
from quantum_torch.qtrainer import QTrainer
from quantum_torch.qoptimizer import QOptimizer
from quantum_torch.qloss import mse

# 2 qubits → 4 classes
model = QuantumLayerModule(2)
trainer = QTrainer(model)
optimizer = QOptimizer(model, lr=0.3)

# Simple dataset (4-class)
dataset = [
    (np.array([0.1, 0.1]), np.array([1, 0, 0, 0])),
    (np.array([0.2, 1.0]), np.array([0, 1, 0, 0])),
    (np.array([2.0, 0.3]), np.array([0, 0, 1, 0])),
    (np.array([2.5, 2.5]), np.array([0, 0, 0, 1])),
]

print("Initial outputs:")
for x, y in dataset:
    print(x, "→", model.forward(x))

# Training loop
for epoch in range(60):
    total_loss = 0
    for x, y in dataset:
        grads = trainer.backward(x, y)
        optimizer.step(grads)
        total_loss += mse(model.forward(x), y)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(dataset):.6f}")

print("\nFinal outputs:")
for x, y in dataset:
    print(x, "→", model.forward(x))
