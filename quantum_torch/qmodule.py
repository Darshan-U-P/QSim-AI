class QModule:
    """
    Base class similar to torch.nn.Module but for Quantum-Torch.
    """

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        """
        Return trainable parameters.
        """
        raise NotImplementedError
