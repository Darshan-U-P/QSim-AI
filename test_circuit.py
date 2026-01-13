from circuit import QuantumCircuit

# Create a 2-qubit quantum circuit
qc = QuantumCircuit(2)

print("Initial state:", qc.state())

# Apply Hadamard to qubit 0
qc.h(0)
print("After H on qubit 0:", qc.state())

# Apply CNOT (entangle)
qc.cnot(0, 1)
print("After CNOT:", qc.state())

print("Probabilities:", qc.probabilities())

print("\nMeasurements (should be only '00' or '11'):")
for _ in range(10):
    print(qc.measure())
