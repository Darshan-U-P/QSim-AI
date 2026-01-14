import torch
import numpy as np


class PBMemory:
    """
    Simple associative vector memory.
    Stores PB embeddings and their labels / concepts.
    """

    def __init__(self):
        self.vectors = []
        self.labels = []

    def store(self, vec, label):
        """
        vec must be a 1-D tensor: [PB_DIM]
        """
        if vec.dim() != 1:
            vec = vec.squeeze(0)

        self.vectors.append(vec.detach().cpu())
        self.labels.append(label)

    def search(self, vec, k=3):
        """
        Returns k most similar stored memories using cosine similarity.
        """
        if len(self.vectors) == 0:
            return [], []

        if vec.dim() != 1:
            vec = vec.squeeze(0)

        vec = vec.detach().cpu()
        sims = []

        for v in self.vectors:
            # v and vec are now both [PB_DIM]
            sim = torch.cosine_similarity(vec, v, dim=0)
            sims.append(sim.item())

        topk = np.argsort(sims)[-k:][::-1]
        return [self.labels[i] for i in topk], [sims[i] for i in topk]
