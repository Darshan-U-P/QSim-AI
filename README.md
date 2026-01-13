# QSim-AI

QSim-AI is a quantum computing simulator written completely from scratch in Python.  
It runs on classical PCs/laptops and is designed specifically for AI/ML/DL research and
hybrid Quantum–Neural architectures.

This project does **not** use PennyLane, Qiskit, Cirq, or any external quantum framework.
Everything is implemented manually for full control, transparency, and research freedom.

The long-term goal is to build a **Quantum–AI engine** that can integrate with PB-ANN and
other custom neural architectures.

---

## Current Project Structure

```
QSim-AI/
│
├── core/
│   ├── __init__.py
│   ├── qubit.py
│   ├── gates.py
│   ├── register.py
│   ├── circuit.py
│   └── quantum_layer.py
│
├── tests/
│   ├── test_qubit.py
│   ├── test_gates.py
│   ├── test_register.py
│   ├── test_entanglement.py
│   ├── test_circuit.py
│   └── test_quantum_layer.py
│
├── docs/
│   └── phase1_core.md
│
├── README.md
└── .gitignore
```

---

## Implemented Features

### Phase 1 – Quantum Core  
✔ Qubit representation  
✔ Probability normalization  
✔ Measurement collapse  
✔ Single-qubit gates (X, Z, H)  
✔ Parameterized rotation gates (RX, RY, RZ)  
✔ Multi-qubit register  
✔ Single-qubit gates on registers  
✔ CNOT gate  
✔ Quantum entanglement  
✔ Bell state generation  

### Phase 2 – Quantum Programming Interface  
✔ QuantumCircuit abstraction  
✔ Clean gate-based programming style  

### Phase 3 – Quantum Neural Layer  
✔ Parameterized circuits  
✔ QuantumLayer abstraction  
✔ Classical → Quantum data encoding  
✔ Quantum probability output  

At this stage, QSim-AI is already capable of behaving like a **Quantum Neural Network layer**.

---

## Example: Entanglement

```python
from core.circuit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cnot(0, 1)

print(qc.state())
print(qc.probabilities())
```

This creates the Bell state:

(|00⟩ + |11⟩) / √2

---

## Example: Quantum Layer (AI Style)

```python
from core.quantum_layer import QuantumLayer
import numpy as np

ql = QuantumLayer(2)
x = np.array([0.5, 1.0])

out = ql.forward(x)
print(out)
print("Sum of probabilities:", out.sum())
```

This behaves like a neural network layer:

```
Input → Quantum Encoding → Quantum Circuit → Measurement → Output Vector
```

---

## Why QSim-AI?

QSim-AI is built for researchers and developers who want:

- Full control over quantum simulation
- No black-box abstractions
- Tight integration with custom AI architectures
- A foundation for hybrid Quantum–PB-ANN systems
- Lightweight, understandable, and extensible code

---

## Roadmap

### Phase 4 – Training Support
- Loss functions
- Gradient estimation (finite difference / parameter-shift rule)
- Weight update mechanisms
- Basic training loops

### Phase 5 – PB-ANN + Quantum Fusion
- QuantumLayer as a feature generator
- Sparse routing
- Hybrid biologically inspired + quantum inspired models

### Phase 6 – Performance & Packaging
- GPU acceleration (optional via NumPy/CuPy)
- Batch circuit execution
- PyPI packaging
- Documentation site

---

## Philosophy

QSim-AI is based on the idea that future intelligence systems will be:

- Biologically inspired  
- Physically grounded  
- Computationally sparse  
- Hybrid by nature  

This project aims to create a bridge between:
**Quantum Physics and Artificial Intelligence.**

---

## Status

The quantum physics core and quantum programming interface are complete.  
The project is now entering the **training and learning phase**, where QSim-AI becomes a
true Quantum Machine Learning framework.
