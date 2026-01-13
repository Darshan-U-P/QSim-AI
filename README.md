# QSim-AI

QSim-AI is a quantum computing simulator written completely from scratch in Python.
It runs on classical computers and is designed for AI/ML/DL research and hybrid
quantum–neural architectures.

This project does NOT use PennyLane, Qiskit, Cirq, or any external quantum framework.

The goal is to build:
- A minimal and transparent quantum simulator
- Fully controllable quantum primitives
- A future Quantum Layer for PB-ANN and AI systems

---

## Implemented Features (Phase 1)

✔ Qubit representation  
✔ Single-qubit gates (X, Z, H)  
✔ Multi-qubit registers  
✔ Single-qubit gates on registers  
✔ CNOT gate  
✔ Quantum entanglement  
✔ Measurement collapse  

This means the core of a real quantum computer is already implemented.

---

## Mathematical Foundation

A quantum state is represented as:

|ψ⟩ = [α₀, α₁, α₂, ..., αₙ]

Where:
- n = 2^number_of_qubits
- ∑ |αᵢ|² = 1

Quantum gates are unitary matrices applied as:

|ψ'⟩ = U |ψ⟩

Measurement collapses the state probabilistically.

---

## Example: Creating Entanglement

```python
qr = QubitRegister(2)
qr.apply_single_gate(H, 0)
qr.apply_cnot(0, 1)
