import numpy as np
import torch
import torch.nn as nn
from quantum_torch.hybrid_trainer import HybridQuantumPBTrainer
from quantum_torch.hybrid_qtrainer import HybridQuantumTrainer

# Setup
NUM_QUBITS = 2
PB_DIM = 32
NUM_CLASSES = 2

model = HybridQuantumPBTrainer(
    n_qubits=NUM_QUBITS,
    pb_dim=PB_DIM,
    num_classes=NUM_CLASSES,
    device="cpu"
)

criterion = nn.CrossEntropyLoss()

# Train Quantum perception
qtrainer = HybridQuantumTrainer(model, eps=1e-3, lr=0.5)

dataset = [
    (np.array([0.1, 0.1]), 0),
    (np.array([0.2, 0.2]), 0),
    (np.array([2.5, 2.5]), 1),
    (np.array([2.2, 2.4]), 1),
]

# --- Capture quantum perception BEFORE training ---
print("Quantum features BEFORE training:")
q_before = model.hybrid.qmodel.forward(np.array([0.1, 0.1]))
print(q_before)

print("\nInitial predictions:")
for x, y in dataset:
    logits = model(x)
    probs = torch.softmax(logits, dim=0)
    print(x, "→", probs.detach().numpy())

# --- Training loop ---
for epoch in range(60):
    total_loss = 0
    for x, y in dataset:
        loss = qtrainer.step(x, y, criterion)
        total_loss += loss

    print(f"Epoch {epoch} | Quantum Loss: {total_loss/len(dataset):.4f}")

# --- Capture quantum perception AFTER training ---
print("\nQuantum features AFTER training:")
q_after = model.hybrid.qmodel.forward(np.array([0.1, 0.1]))
print(q_after)

print("\nFinal predictions:")
for x, y in dataset:
    logits = model(x)
    probs = torch.softmax(logits, dim=0)
    print(x, "→", probs.detach().numpy())
