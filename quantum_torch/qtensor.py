import numpy as np

class QTensor:
    """
    Quantum tensor = quantum state container.
    Holds amplitudes or probability vector.
    """

    def __init__(self, data):
        self.data = np.array(data, dtype=np.complex128)

    def normalize(self):
        norm = np.linalg.norm(self.data)
        if norm != 0:
            self.data = self.data / norm

    def probabilities(self):
        return np.abs(self.data) ** 2

    def __repr__(self):
        return f"QTensor(shape={self.data.shape})"
