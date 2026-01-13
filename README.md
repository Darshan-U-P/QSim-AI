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
QSim-AI/
│
├── core/
│ ├── init.py
│ ├── qubit.py
│ ├── gates.py
│ ├── register.py
│ ├── circuit.py
│ └── quantum_layer.py
│
├── tests/
│ ├── test_qubit.py
│ ├── test_gates.py
│ ├── test_register.py
│ ├── test_entanglement.py
│ ├── test_circuit.py
│ └── test_quantum_layer.py
│
├── docs/
│ └── phase1_core.md
│
├── README.md
└── .gitignore


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
