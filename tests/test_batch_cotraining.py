import numpy as np
import torch
import torch.nn as nn
from quantum_torch.hybrid_trainer import HybridQuantumPBTrainer
from quantum_torch.hybrid_batch_cotrainer import HybridBatchCoTrainer

# ===== INPUT PADDER =====
def pad_input(x, n_qubits):
    if len(x) < n_qubits:
        x = np.concatenate([x, np.zeros(n_qubits - len(x))])
    return x


# ===== CONFIG =====
NUM_QUBITS = 3          # ⬅️ Scaled from 2 → 3 (Perception resolution increased)
PB_DIM = 32
NUM_CLASSES = 2

# ===== MODEL =====
model = HybridQuantumPBTrainer(
    n_qubits=NUM_QUBITS,
    pb_dim=PB_DIM,
    num_classes=NUM_CLASSES,
    device="cpu"
)

criterion = nn.CrossEntropyLoss()

trainer = HybridBatchCoTrainer(
    model,
    eps=5e-4,        # Quantum gradient resolution
    q_lr=0.6,        # Quantum learning rate
    pb_lr=1e-3       # PB learning rate
)

# ===== DATASET =====
# Still 2D semantic data → padded into 3D quantum perception
raw_dataset = [
    (np.array([0.1, 0.1]), 0),
    (np.array([0.2, 0.2]), 0),
    (np.array([2.5, 2.5]), 1),
    (np.array([2.2, 2.4]), 1),
]

dataset = [(pad_input(x, NUM_QUBITS), y) for x, y in raw_dataset]

# ===== INITIAL STATE =====
print("\nInitial predictions:")
for x, y in dataset:
    probs = torch.softmax(model(x), dim=0)
    print(x, "→", probs.detach().numpy())

# ===== TRAINING LOOP =====
for epoch in range(120):
    loss = trainer.step(dataset, criterion)
    print(f"Epoch {epoch} | Batch Co-Learning Loss: {loss:.4f}")

# ===== FINAL STATE =====
print("\nFinal predictions:")
for x, y in dataset:
    probs = torch.softmax(model(x), dim=0)
    print(x, "→", probs.detach().numpy())
