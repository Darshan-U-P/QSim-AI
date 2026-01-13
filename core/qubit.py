import numpy as np

class Qubit:
    def __init__(self, alpha=1+0j, beta=0+0j):
        self.state = np.array([alpha, beta], dtype=complex)
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm == 0:
            raise ValueError("Invalid qubit state")
        self.state = self.state / norm

    def apply_gate(self, gate):
        self.state = gate @ self.state
        self.normalize()

    def probabilities(self):
        return np.abs(self.state) ** 2

    def measure(self):
        probs = self.probabilities()
        return np.random.choice([0, 1], p=probs)

    def __repr__(self):
        return f"Qubit(state={self.state})"
