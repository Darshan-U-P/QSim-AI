from register import QubitRegister
import gates

class QuantumCircuit:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.register = QubitRegister(n_qubits)

    # Single-qubit gates
    def h(self, qubit):
        self.register.apply_single_gate(gates.H, qubit)

    def x(self, qubit):
        self.register.apply_single_gate(gates.X, qubit)

    def z(self, qubit):
        self.register.apply_single_gate(gates.Z, qubit)

    # Two-qubit gate
    def cnot(self, control, target):
        self.register.apply_cnot(control, target)

    # Measurement
    def measure(self):
        return self.register.measure()

    def probabilities(self):
        return self.register.probabilities()

    def state(self):
        return self.register.state
