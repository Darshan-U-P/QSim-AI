class QOptimizer:
    """
    Simple gradient descent optimizer for Quantum-Torch
    """

    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr

    def step(self, grads):
        params = self.model.parameters()
        params -= self.lr * grads
