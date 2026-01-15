import numpy as np
import torch
import torch.nn as nn
from quantum_torch.hybrid_trainer import HybridQuantumPBTrainer
from quantum_torch.hybrid_batch_cotrainer import HybridBatchCoTrainer
from quantum_torch.language_head import LanguageHead

# =====================
# CONFIG
# =====================
NUM_QUBITS = 5
PB_DIM = 128
VOCAB = ["small", "big"]
vocab_size = len(VOCAB)

# =====================
# MODEL
# =====================
model = HybridQuantumPBTrainer(
    n_qubits=NUM_QUBITS,
    pb_dim=PB_DIM,
    num_classes=2,
    device="cpu"
)

trainer = HybridBatchCoTrainer(
    model,
    eps=5e-4,
    q_lr=0.6,
    pb_lr=1e-3
)

language = LanguageHead(PB_DIM, vocab_size)
optimizer = torch.optim.Adam(language.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# =====================
# SYMBOL DATASET
# =====================
# PB perception → word
# Pad inputs to match 5 qubits
dataset = [
    (np.array([0.1, 0.1, 0, 0, 0]), 0),  # "small"
    (np.array([0.2, 0.2, 0, 0, 0]), 0),  # "small"
    (np.array([2.5, 2.5, 0, 0, 0]), 1),  # "big"
    (np.array([2.2, 2.4, 0, 0, 0]), 1),  # "big"
]

# =====================
# TRAIN LANGUAGE
# =====================
print("Training Symbol Grounding...")
for epoch in range(80):
    total_loss = 0
    for x, token in dataset:
        # 1. Perception (Quantum -> PB)
        pb = trainer.embed(x)     
        
        # 2. Language Generation
        logits = language(pb)
        
        # 3. FORCE SHAPE (The Fix)
        # Ensure it is exactly (Batch=1, Classes=2)
        logits = logits.view(1, -1) 
        
        targets = torch.tensor([token], dtype=torch.long)
        
        loss = criterion(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Symbol Loss: {total_loss/len(dataset):.4f}")

# =====================
# TEST
# =====================
print("\nTesting Grounding:")
test_inputs = [
    (np.array([0.15, 0.15, 0, 0, 0]), "small"),
    (np.array([2.3, 2.3, 0, 0, 0]), "big")
]

for x, label in test_inputs:
    pb = trainer.embed(x)
    logits = language(pb)
    
    # Force shape for inference too
    logits = logits.view(1, -1)
    
    probs = torch.softmax(logits, dim=1)
    idx = torch.argmax(probs).item()
    
    print(f"Input {x[:2]}... → Predicted: {VOCAB[idx]} ({probs[0][idx]:.4f}) | Truth: {label}")