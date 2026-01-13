import numpy as np
from core.quantum_layer import QuantumLayer
from core.trainer import QuantumTrainer

# Convert 4 quantum states -> 2 classes
def quantum_to_class_probs(p):
    # Sum probabilities of states to form class probabilities
    p0 = p[0] + p[1]   # |00>, |01> -> Class 0
    p1 = p[2] + p[3]   # |10>, |11> -> Class 1
    return np.array([p0, p1])

# 2-qubit model
model = QuantumLayer(2)

# Bad initialization
model.weights[:] = np.random.uniform(2.5, 3.0, size=2)

# Initialize trainer with the transform function
trainer = QuantumTrainer(
    model,
    lr=0.2,
    output_transform=quantum_to_class_probs
)

x = np.array([0.4, 1.1])
y_target = np.array([1.0, 0.0])

print("Initial quantum output:", model.forward(x))
print("Initial class output:", quantum_to_class_probs(model.forward(x)))

for epoch in range(40):
    loss = trainer.step(x, y_target)
    
    if epoch % 5 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss:.6f} | "
            f"Class output: {quantum_to_class_probs(model.forward(x))}"
        )

print("\nFinal quantum output:", model.forward(x))
print("Final class output:", quantum_to_class_probs(model.forward(x)))