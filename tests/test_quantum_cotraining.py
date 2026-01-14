import numpy as np
import torch
import torch.nn as nn
from quantum_torch.hybrid_trainer import HybridQuantumPBTrainer
from quantum_torch.hybrid_cotrainer import HybridCoTrainer

model = HybridQuantumPBTrainer(
    n_qubits=2,
    pb_dim=32,
    num_classes=2,
    device="cpu"
)

criterion = nn.CrossEntropyLoss()
trainer = HybridCoTrainer(model, quantum_eps=1e-3, quantum_lr=0.3, pb_lr=1e-3)

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

for epoch in range(60):
    total_loss = 0
    for x, y in dataset:
        # Quantum update
        q_loss = trainer.quantum_step(x, y, criterion)
        # PB update
        pb_loss = trainer.pb_step(x, y, criterion)

        total_loss += (q_loss + pb_loss)

    print(f"Epoch {epoch} | Co-Learning Loss: {total_loss/len(dataset):.4f}")

print("\nFinal predictions:")
for x, y in dataset:
    logits = model(x)
    probs = torch.softmax(logits, dim=0)
    print(x, "→", probs.detach().numpy())
