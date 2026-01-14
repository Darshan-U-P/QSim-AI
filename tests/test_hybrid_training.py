import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from quantum_torch.hybrid_trainer import HybridQuantumPBTrainer

# Configuration
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
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Simple dataset
dataset = [
    (np.array([0.1, 0.1]), 0),
    (np.array([0.2, 0.2]), 0),
    (np.array([2.5, 2.5]), 1),
    (np.array([2.2, 2.4]), 1),
]

print("Initial predictions:")
for x, y in dataset:
    logits = model(x)
    probs = torch.softmax(logits, dim=0)
    print(x, "→", probs.detach().numpy())

# Training
for epoch in range(50):
    total_loss = 0
    for x, y in dataset:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.unsqueeze(0), torch.tensor([y]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(dataset):.4f}")

print("\nFinal predictions:")
for x, y in dataset:
    logits = model(x)
    probs = torch.softmax(logits, dim=0)
    print(x, "→", probs.detach().numpy())
