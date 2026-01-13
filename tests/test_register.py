from register import QubitRegister
import gates

# Create a 2-qubit register initialized to |00>
qr = QubitRegister(2)

print("Initial register state:")
print(qr.state)
print("Probabilities:", qr.probabilities())

# Apply Hadamard gate to qubit 0
qr.apply_single_gate(gates.H, target=0)

print("\nAfter applying Hadamard on qubit 0:")
print(qr.state)
print("Probabilities:", qr.probabilities())

# Measure multiple times
print("\nMeasurements (should be only '00' or '10'):")
for _ in range(10):
    print(qr.measure())
