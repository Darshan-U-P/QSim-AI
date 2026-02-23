# QSim-AI

## Quantum--Neural Cognitive Architecture Framework

QSim-AI is a self-contained Quantum--Neural research framework
implemented entirely from first principles in Python.\
It runs on classical computing hardware and is designed for structured
experimentation in:

-   Quantum Machine Learning (QML)
-   Hybrid quantum--classical computation
-   Sparse neural architectures
-   Representation learning systems

The framework does not rely on external quantum SDKs such as Qiskit,
PennyLane, or Cirq.\
All simulation, parameterization, and optimization logic is implemented
manually to ensure full mathematical transparency and architectural
control.

------------------------------------------------------------------------

# Repository Structure

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
    ├── quantum_torch/
    │   ├── qtensor.py
    │   ├── qmodule.py
    │   ├── qlayer.py
    │   ├── qloss.py
    │   ├── qoptimizer.py
    │   ├── qtrainer.py
    │   ├── bridge.py
    │   ├── hybrid.py
    │   ├── hybrid_qtrainer.py
    │   └── hybrid_cotrainer.py
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
    │   ├── test_hybrid_training.py
    │   ├── test_quantum_colearning.py
    │   └── test_quantum_cotraining.py
    │
    ├── docs/
    │   └── phase1_core.md
    │
    ├── README.md
    └── .gitignore

------------------------------------------------------------------------

# Core Capabilities

## Quantum Simulation

-   Qubit state representation
-   Probability normalization
-   Measurement collapse
-   Single-qubit gates (X, Z, H)
-   Parameterized rotations (RX, RY, RZ)
-   Multi-qubit registers
-   CNOT gate and entanglement
-   Bell state generation

## Quantum Machine Learning

-   Parameterized quantum circuits
-   Classical-to-quantum encoding
-   Measurement-based probability outputs
-   Finite-difference gradient estimation
-   Gradient descent optimization
-   Binary and multi-class classification
-   Batch training

## Quantum-Torch Framework

-   QTensor (quantum state container)
-   QModule (layer abstraction)
-   QOptimizer (parameter updates)
-   QTrainer (backward engine)
-   QLoss (loss functions)
-   Standalone quantum training without PyTorch

## Hybrid Quantum--Neural Integration

-   Quantum feature extraction
-   Sparse neural module integration
-   Joint quantum--neural co-training
-   Closed-loop gradient optimization

------------------------------------------------------------------------

# Architectural Overview

Pipeline:

Input\
↓\
Quantum Layer (Probabilistic Encoding)\
↓\
Quantum Probability Features\
↓\
Sparse Neural Module\
↓\
Decision / Routing / Language Layer

This framework is designed for research in hybrid quantum--neural
representation learning and structured intelligence systems.

------------------------------------------------------------------------

# Research Positioning

QSim-AI is positioned at the intersection of:

-   Quantum Machine Learning
-   Sparse Neural Networks
-   Hybrid Computational Architectures
-   Structured Representation Learning

It is intended as a controlled experimental environment for evaluating
quantum-enhanced feature representations in advanced AI systems.

------------------------------------------------------------------------

# Status

-   Quantum simulation engine: Completed
-   Trainable quantum layers: Completed
-   Hybrid quantum--neural integration: Completed
-   Generative layer integration: In progress

------------------------------------------------------------------------

# License

Research and experimental use.
