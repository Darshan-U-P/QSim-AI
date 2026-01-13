import numpy as np
from core.quantum_layer import QuantumLayer
from core.trainer import QuantumTrainer

model = QuantumLayer(1)
trainer = QuantumTrainer(model, lr=0.3)

# Force bad initial weights
model.weights[:] = np.random.uniform(2.5, 3.0, size=1)

x = np.array([0.2])
y_target = np.array([1.0, 0.0])

print("Initial output:", model.forward(x))

for epoch in range(40):
    loss = trainer.step(x, y_target)
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f} | Output: {model.forward(x)}")

print("\nFinal output:", model.forward(x))
