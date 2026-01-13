from qubit import Qubit
import gates

# Start with |0>
q = Qubit(1, 0)
print("Initial:", q, q.probabilities())

# Apply X gate → should become |1>
q.apply_gate(gates.X)
print("After X:", q, q.probabilities())

# Apply Z gate → phase flip (|1> stays |1>, but phase changes)
q.apply_gate(gates.Z)
print("After Z:", q, q.probabilities())

# Reset to |0>
q = Qubit(1, 0)

# Apply H → superposition
q.apply_gate(gates.H)
print("After H:", q, q.probabilities())

# Measure many times
print("Measurements after H:")
for _ in range(10):
    print(q.measure())
