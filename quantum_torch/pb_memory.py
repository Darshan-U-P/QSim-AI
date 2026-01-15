import numpy as np
import torch
import torch.nn.functional as F

class PBMemory:
    """
    Simple associative vector memory.
    Stores PB embeddings and their labels / concepts.
    """

    def __init__(self):
        self.vectors = []
        self.labels = []

    def store(self, vec, label):
        # Ensure 1D tensor
        if vec.dim() > 1:
            vec = vec.squeeze()
        self.vectors.append(vec.detach().cpu())
        self.labels.append(label)

    def search(self, vec, k=3):
        """
        Returns k most similar stored memories using cosine similarity.
        """
        if len(self.vectors) == 0:
            return [], []

        # Ensure 1D tensor
        if vec.dim() > 1:
            vec = vec.squeeze()

        vec = vec.detach().cpu()
        sims = []

        for v in self.vectors:
            # Force both to shape [1, PB_DIM]
            sim = F.cosine_similarity(
                vec.unsqueeze(0),
                v.unsqueeze(0),
                dim=1
            )
            sims.append(sim.item())  # Now this is a scalar

        topk = np.argsort(sims)[-k:][::-1]

        return [self.labels[i] for i in topk], [sims[i] for i in topk]
