import torch
import torch.nn as nn

class LanguageHead(nn.Module):
    """
    Maps PB cognitive vectors to discrete symbols (tokens / words).
    This is Symbol Grounding.
    """
    def __init__(self, pb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(pb_dim, vocab_size)

    def forward(self, pb):
        return self.fc(pb)
