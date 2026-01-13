import numpy as np
from functools import reduce

class QubitRegister:
    def __init__(self, n):
        self.n = n
        self.dim = 2 ** n
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0  # |00...0>

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm == 0:
            raise ValueError("Invalid quantum state")
        self.state = self.state / norm

    def probabilities(self):
        return np.abs(self.state) ** 2

    def measure(self):
        probs = self.probabilities()
        index = np.random.choice(range(self.dim), p=probs)
        return format(index, f'0{self.n}b')

    def apply_single_gate(self, gate, target):
        ops = []
        for i in range(self.n):
            if i == target:
                ops.append(gate)
            else:
                ops.append(np.eye(2))
        U = reduce(np.kron, ops)
        self.state = U @ self.state
        self.normalize()

    # ADD THIS METHOD HERE
    def apply_cnot(self, control, target):
        """
        Apply a CNOT gate with given control and target qubits.
        """
        new_state = np.zeros_like(self.state)

        for i in range(self.dim):
            # Binary representation of basis index
            bits = list(format(i, f'0{self.n}b'))

            # If control qubit is 1, flip target qubit
            if bits[control] == '1':
                bits[target] = '0' if bits[target] == '1' else '1'

            # New index after CNOT
            new_index = int("".join(bits), 2)

            # Move amplitude
            new_state[new_index] += self.state[i]

        self.state = new_state
        self.normalize()

    def __repr__(self):
        return f"QubitRegister({self.n} qubits, state={self.state})"
