
---

Now update **docs/phase1_core.md** to this:

```markdown
# Phase 1 – Quantum Core Implementation

Phase 1 establishes the complete mathematical and physical foundation of QSim-AI.

This phase focuses purely on quantum mechanics simulation, without AI or learning.

---

## 1. Qubit

A qubit is represented as a complex vector:

|ψ⟩ = [ α , β ]

Where:

|α|² + |β|² = 1  

Implemented features:
- Complex amplitude representation
- Automatic normalization
- Probability extraction
- Measurement collapse

---

## 2. Single-Qubit Gates

Implemented:

| Gate | Meaning |
|------|-------|
X | Bit flip (Quantum NOT)
Z | Phase flip
H | Superposition creator
RX(θ) | Rotation around X-axis
RY(θ) | Rotation around Y-axis
RZ(θ) | Rotation around Z-axis

All gates are unitary matrices:

|ψ'⟩ = U |ψ⟩

---

## 3. Multi-Qubit Register

A system of *n* qubits is represented by a state vector of size:

\[
2^n
\]

Implemented:
- N-qubit state memory
- Tensor product construction
- Global normalization
- Full measurement collapse

---

## 4. Local Gate Application

Single-qubit gates can be applied inside a multi-qubit system using:

\[
U = I ⊗ I ⊗ ... ⊗ G ⊗ ... ⊗ I
\]

This allows:
- Independent control of each qubit
- Realistic quantum behavior

---

## 5. Controlled Operations (CNOT)

CNOT is defined as:

If control = 1 → flip target  
If control = 0 → do nothing  

Truth table:

| Input | Output |
|------|------|
00 | 00  
01 | 01  
10 | 11  
11 | 10  

This gate is the backbone of quantum entanglement.

---

## 6. Entanglement

Using:

1. Hadamard on qubit 0  
2. CNOT(0, 1)

We create:

\[
(|00⟩ + |11⟩) / \sqrt{2}
\]

Which shows:
- Non-classical correlation
- Linked measurement collapse
- True quantum behavior

---

## 7. Circuit Abstraction

QuantumCircuit provides:

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cnot(0,1)
