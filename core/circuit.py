from core.register import QubitRegister
import core.gates as gates


class QuantumCircuit:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.register = QubitRegister(n_qubits)

    # -------------------------
    # Fixed Gates
    # -------------------------

    def h(self, qubit):
        self.register.apply_single_gate(gates.H, qubit)

    def x(self, qubit):
        self.register.apply_single_gate(gates.X, qubit)

    def z(self, qubit):
        self.register.apply_single_gate(gates.Z, qubit)

    # -------------------------
    # Parameterized Gates
    # -------------------------

    def rx(self, qubit, theta):
        self.register.apply_single_gate(gates.RX(theta), qubit)

    def ry(self, qubit, theta):
        self.register.apply_single_gate(gates.RY(theta), qubit)

    def rz(self, qubit, theta):
        self.register.apply_single_gate(gates.RZ(theta), qubit)

    # -------------------------
    # Two-Qubit Gate
    # -------------------------

    def cnot(self, control, target):
        self.register.apply_cnot(control, target)

    # -------------------------
    # Output / Measurement
    # -------------------------

    def measure(self):
        return self.register.measure()

    def probabilities(self):
        return self.register.probabilities()

    def state(self):
        return self.register.state
