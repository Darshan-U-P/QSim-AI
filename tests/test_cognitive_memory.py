import numpy as np
import torch
from quantum_torch.hybrid_trainer import HybridQuantumPBTrainer
from quantum_torch.hybrid_batch_cotrainer import HybridBatchCoTrainer
from quantum_torch.pb_memory import PBMemory
import torch.nn as nn

# ===== CONFIG =====
NUM_QUBITS = 5
PB_DIM = 128
NUM_CLASSES = 2

# ===== MODEL =====
model = HybridQuantumPBTrainer(
    n_qubits=NUM_QUBITS,
    pb_dim=PB_DIM,
    num_classes=NUM_CLASSES,
    device="cpu"
)

trainer = HybridBatchCoTrainer(
    model,
    eps=5e-4,
    q_lr=0.6,
    pb_lr=1e-3
)

criterion = nn.CrossEntropyLoss()

# ===== DATASET =====
dataset = [
    (np.array([0.1, 0.1, 0, 0, 0]), 0),
    (np.array([0.2, 0.2, 0, 0, 0]), 0),
    (np.array([2.5, 2.5, 0, 0, 0]), 1),
    (np.array([2.2, 2.4, 0, 0, 0]), 1),
]

# ===== TRAIN QUICKLY =====
print("Training model...")
for epoch in range(40):
    loss = trainer.step(dataset, criterion)
    print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ===== MEMORY =====
memory = PBMemory()

print("\nStoring experiences in memory:")
for x, label in dataset:
    pb, q = trainer.encode(x)          # encode → (pb, q)
    memory.store(pb, f"class_{label}")
    print(f"Stored: {x} → class_{label}")

# ===== QUERY MEMORY =====
print("\nQuerying memory:")
test_input = np.array([0.15, 0.15, 0, 0, 0])

pb_query = trainer.embed(test_input)   # embed → pb only
labels, scores = memory.search(pb_query, k=3)

print("Input:", test_input)
print("Closest memories:")
for l, s in zip(labels, scores):
    print(f"  {l} | similarity: {s:.4f}")
