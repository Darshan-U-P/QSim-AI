import torch
import numpy as np


class CognitiveGenerator:
    """
    Memory-augmented language generator.

    PB cognition → Memory recall → Fusion → Language generation
    """

    def __init__(self, pb_text_head, memory, alpha=0.7, device="cpu"):
        self.text_head = pb_text_head
        self.memory = memory
        self.alpha = alpha
        self.device = device

    def fuse_pb(self, pb, recalled_pbs):
        if len(recalled_pbs) == 0:
            return pb

        mem_stack = torch.stack(recalled_pbs, dim=0)  # (K, PB_DIM)
        mem_mean = mem_stack.mean(dim=0)              # (PB_DIM)

        fused = self.alpha * pb + (1 - self.alpha) * mem_mean
        return fused

    def generate(self, pb, k=3, max_len=6):
        """
        pb : (PB_DIM,) tensor
        """
        if isinstance(pb, np.ndarray):
            pb = torch.tensor(pb, dtype=torch.float32)

        pb = pb.to(self.device)

        labels, scores, recalled_pbs = self.memory.search_with_vectors(pb, k=k)
        recalled_pbs = [v.to(self.device) for v in recalled_pbs]

        fused_pb = self.fuse_pb(pb, recalled_pbs)

        # IMPORTANT: fused_pb must be (PB_DIM,) not batched
        sentence = self.text_head.generate(fused_pb, max_len=max_len)

        return sentence, labels, scores
