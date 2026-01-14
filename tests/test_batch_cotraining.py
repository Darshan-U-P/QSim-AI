import numpy as np
import torch
import torch.nn as nn
from quantum_torch.hybrid_trainer import HybridQuantumPBTrainer
from quantum_torch.hybrid_batch_cotrainer import HybridBatchCoTrainer

model = HybridQuantumPBTrainer(
    n_qubits=2,
    pb_dim=32,
    num_classes=2,
    device="cpu"
)

criterion = nn.CrossEntropyLoss()
trainer = HybridBatchCoTrainer(model, eps=5e-4, q_lr=0.6, pb_lr=1e-3)

dataset = [
    (np.array([0.1, 0.1]), 0),
    (np.array([0.2, 0.2]), 0),
    (np.array([2.5, 2.5]), 1),
    (np.array([2.2, 2.4]), 1),
]

print("Initial predictions:")
for x, y in dataset:
    p = torch.softmax(model(x), dim=0)
    print(x, "→", p.detach().numpy())

for epoch in range(120):
    loss = trainer.step(dataset, criterion)
    print(f"Epoch {epoch} | Batch Co-Learning Loss: {loss:.4f}")

print("\nFinal predictions:")
for x, y in dataset:
    p = torch.softmax(model(x), dim=0)
    print(x, "→", p.detach().numpy())
