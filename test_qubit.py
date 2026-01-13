from qubit import Qubit

# |0> state
q0 = Qubit(1, 0)
print("Qubit |0>:", q0)
print("Probabilities:", q0.probabilities())

# |1> state
q1 = Qubit(0, 1)
print("Qubit |1>:", q1)
print("Probabilities:", q1.probabilities())

# Superposition manually
q = Qubit(1, 1)
print("Superposition Qubit:", q)
print("Probabilities:", q.probabilities())

# Measurement test
for i in range(10):
    print("Measurement:", q.measure())
