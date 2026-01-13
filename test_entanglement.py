from register import QubitRegister
import gates

# Create 2-qubit register |00>
qr = QubitRegister(2)
print("Initial:", qr.state)

# Step 1: Put first qubit in superposition
qr.apply_single_gate(gates.H, target=0)
print("After H on qubit 0:", qr.state)

# Step 2: Apply CNOT (entanglement)
qr.apply_cnot(control=0, target=1)
print("After CNOT (Entangled state):", qr.state)

print("Probabilities:", qr.probabilities())

# Measure many times
print("\nMeasurements (should be only '00' or '11'):")
for _ in range(10):
    print(qr.measure())
