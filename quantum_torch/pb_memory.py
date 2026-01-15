import torch
import numpy as np


class PBMemory:
    """
    Simple associative memory storing PB embeddings and their text labels.
    """

    def __init__(self):
        self.vectors = []
        self.labels = []

    def store(self, vec, label):
        # vec must be [PB_DIM]
        vec = vec.detach().cpu().view(-1)
        self.vectors.append(vec)
        self.labels.append(label)

    def search(self, vec, k=3):
        labels, scores, _ = self.search_with_vectors(vec, k)
        return labels, scores

    def search_with_vectors(self, vec, k=3):
        """
        Returns:
        - labels: recalled text labels
        - scores: similarity scores
        - recalled_pbs: actual PB vectors that were recalled
        """
        if len(self.vectors) == 0:
            return [], [], []

        vec = vec.detach().cpu().view(-1)

        sims = []
        for v in self.vectors:
            v = v.view(-1)
            sim = torch.dot(vec, v) / (torch.norm(vec) * torch.norm(v) + 1e-8)
            sims.append(sim.item())

        topk = np.argsort(sims)[-k:][::-1]

        recalled_labels = [self.labels[i] for i in topk]
        recalled_scores = [sims[i] for i in topk]
        recalled_vectors = [self.vectors[i] for i in topk]

        return recalled_labels, recalled_scores, recalled_vectors
