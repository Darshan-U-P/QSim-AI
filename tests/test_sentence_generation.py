import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from quantum_torch.pb_text_head import PBTextHead

# ============================
# Vocabulary
# ============================
vocab = ["<pad>", "<start>", "<end>", "this", "is", "small", "big"]
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}

VOCAB_SIZE = len(vocab)
PB_DIM = 128
DEVICE = "cpu"

# ============================
# Fake PB embeddings
# ============================
pb_small = torch.randn(PB_DIM)
pb_big   = torch.randn(PB_DIM)

# Sentences:
# <start> this is small <end>
# <start> this is big <end>

targets = [
    (pb_small, [stoi["<start>"], stoi["this"], stoi["is"], stoi["small"], stoi["<end>"]]),
    (pb_big,   [stoi["<start>"], stoi["this"], stoi["is"], stoi["big"],   stoi["<end>"]]),
]

# ============================
# Model
# ============================
model = PBTextHead(PB_DIM, VOCAB_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

# ============================
# Training
# ============================
print("Training Sentence Generator...")

for epoch in range(500):
    total_loss = 0
    for pb, seq in targets:
        pb = pb.to(DEVICE)
        seq = torch.tensor(seq, device=DEVICE)

        inputs  = seq[:-1]
        targets_tokens = seq[1:]

        logits = model(pb, inputs)
        loss = criterion(logits, targets_tokens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

# ============================
# Generation
# ============================
def decode(tokens):
    words = []
    for t in tokens:
        if t == stoi["<end>"]:
            break
        words.append(itos[t])
    return " ".join(words)

print("\nTesting Generation:")

out_small = model.generate(pb_small, stoi["<start>"], max_len=6)
print("PB → small:", decode(out_small))

out_big = model.generate(pb_big, stoi["<start>"], max_len=6)
print("PB → big:", decode(out_big))
