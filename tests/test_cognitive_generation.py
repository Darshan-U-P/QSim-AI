import numpy as np
import torch
import torch.nn as nn

from quantum_torch.hybrid_trainer import HybridQuantumPBTrainer
from quantum_torch.hybrid_batch_cotrainer import HybridBatchCoTrainer
from quantum_torch.pb_memory import PBMemory
from quantum_torch.pb_text_head import PBTextHead
from quantum_torch.cognitive_generator import CognitiveGenerator


# ===== CONFIG =====
NUM_QUBITS = 5
PB_DIM = 128
VOCAB = ["<bos>", "<eos>", "this", "is", "small", "big"]


# ===== MODEL =====
model = HybridQuantumPBTrainer(
    n_qubits=NUM_QUBITS,
    pb_dim=PB_DIM,
    num_classes=2,
    device="cpu"
)

trainer = HybridBatchCoTrainer(model)
memory = PBMemory()
text_head = PBTextHead(pb_dim=PB_DIM, vocab=VOCAB)
cognitive = CognitiveGenerator(text_head, memory)

criterion = nn.CrossEntropyLoss()

# ===== TRAIN PB MODEL QUICKLY =====
dataset = [
    (np.array([0.1, 0.1, 0, 0, 0]), 0),
    (np.array([0.2, 0.2, 0, 0, 0]), 0),
    (np.array([2.5, 2.5, 0, 0, 0]), 1),
    (np.array([2.2, 2.4, 0, 0, 0]), 1),
]

print("Training cognitive core...")
for epoch in range(30):
    loss = trainer.step(dataset, criterion)
    print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ===== STORE MEMORY =====
print("\nStoring memory:")
for x, label in dataset:
    pb = trainer.embed(x)
    text = "this is small" if label == 0 else "this is big"
    memory.store(pb, text)
    print(f"Stored: {x} → '{text}'")

# ===== TEST GENERATION =====
print("\nTesting Cognitive Generation:")
test_inputs = [
    np.array([0.15, 0.15, 0, 0, 0]),
    np.array([2.4, 2.4, 0, 0, 0]),
]

for x in test_inputs:
    pb = trainer.embed(x)
    sentence, labels, scores = cognitive.generate(pb, k=2)

    print("\nInput:", x)
    print("Recalled memory:", labels)
    print("Generated:", sentence)
