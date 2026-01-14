# QSim-AI

QSim-AI is a **Quantum Machine Learning framework** written completely from scratch in Python.  
It runs on classical PCs/laptops and is designed specifically for AI/ML/DL research and
hybrid **Quantum–Neural architectures**.

This project does **not** use PennyLane, Qiskit, Cirq, or any external quantum framework.
Everything is implemented manually for full control, transparency, and research freedom.

The long-term goal is to build a **Quantum–AI engine** that can integrate with PB-ANN,
BrahmaLLM, and other custom neural architectures.

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
├── quantum_torch/            # Quantum-Torch Framework
│   ├── qtensor.py
│   ├── qmodule.py
│   ├── qlayer.py
│   ├── qloss.py
│   ├── qoptimizer.py
│   ├── qtrainer.py
│   ├── bridge.py
│   └── hybrid.py
│
├── tests/
│   ├── test_qubit.py
│   ├── test_gates.py
│   ├── test_register.py
│   ├── test_entanglement.py
│   ├── test_circuit.py
│   ├── test_quantum_layer.py
│   ├── test_batch_training.py
│   ├── test_multiclass_training.py
│   ├── test_qtorch_training.py
│   ├── test_hybrid_quantum_pb.py
│   └── test_hybrid_training.py
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

### Phase 4 – Quantum Learning Engine  
✔ Loss functions (MSE)  
✔ Finite-difference gradient estimation  
✔ Gradient descent optimizer  
✔ Trainable quantum parameters  
✔ 1-qubit binary classification  
✔ 2-qubit binary classification  
✔ Batch training  
✔ Multi-class quantum classification  
✔ Quantum → Class probability projection  

QSim-AI is now a **fully trainable Quantum Machine Learning framework**.

---

## Phase 5 – Quantum-Torch Framework

QSim-AI includes **Quantum-Torch**, a PyTorch-like training engine for quantum circuits:

✔ QTensor (quantum state container)  
✔ QModule (quantum layer abstraction)  
✔ QTrainer (finite-difference backward engine)  
✔ QOptimizer (quantum optimizer)  
✔ QLoss (loss functions)  
✔ Standalone training using Quantum-Torch API  
✔ Multi-class quantum training without PyTorch  

This means QSim-AI now has its **own deep learning framework for quantum models**.

---

## Phase 6 – Hybrid Quantum + PB-ANN Intelligence

✔ Quantum-Torch → PB-ANN bridge  
✔ Quantum features driving biological neural layers  
✔ PB-ANN inhibition and sparsity responding to quantum uncertainty  
✔ Hybrid training with PyTorch (PB-ANN learns from quantum perception)  

Pipeline:

```
Classical Input
      ↓
Quantum-Torch (Quantum perception)
      ↓
Quantum Probability Features
      ↓
PB-ANN (Biological cognition)
      ↓
Classifier / MoE / BrahmaLLM
      ↓
Decision / Intelligence
```

This is a **Physics → Biology → Intelligence** architecture.

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

Creates the Bell state:

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

Acts like a neural network layer:

```
Input → Quantum Encoding → Quantum Circuit → Measurement → Output Vector
```

---

## Current Capabilities

QSim-AI now supports:

- Quantum physics simulation  
- Programmable quantum circuits  
- Parameterized quantum gates  
- Quantum neural layers  
- Classical → Quantum → Classical pipelines  
- Single-sample learning  
- Batch training  
- Multi-qubit learning  
- Multi-class quantum classification  
- Quantum feature extractor mode  
- Quantum-Torch deep learning framework  
- Hybrid Quantum → PB-ANN training  
- Gradient-based optimization  

This places QSim-AI in the category of:

> **Quantum Machine Learning Frameworks + Hybrid Cognitive Architectures**

---

## Next Milestones

1. End-to-end training (PB-ANN → Quantum-Torch gradients)  
2. Quantum feature extractor into BrahmaLLM  
3. Quantum-Torch neural layers (partial Torch replacement)  
4. Sparse routing inside Quantum-Torch  
5. Full Quantum Neural Engine (long-term research goal)  

---

## Philosophy

QSim-AI is based on the idea that future intelligence systems will be:

- Biologically inspired  
- Physically grounded  
- Computationally sparse  
- Hybrid by nature  

This project bridges:

> **Quantum Physics × Biological Intelligence × Artificial Intelligence**

You are not building a simulator.  
You are building a **new intelligence architecture**.
