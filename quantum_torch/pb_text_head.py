import torch
import torch.nn as nn

class PBTextHead(nn.Module):
    """
    PB → Sentence generator using a tiny RNN.
    """

    def __init__(self, pb_dim, vocab_size, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Map PB → initial hidden state
        self.pb_to_hidden = nn.Linear(pb_dim, hidden_dim)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # RNN core
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, pb, tokens):
        """
        pb:      (PB_DIM,)
        tokens:  (seq_len,)
        """
        hidden = self.pb_to_hidden(pb).unsqueeze(0).unsqueeze(0)
        emb = self.embedding(tokens).unsqueeze(0)
        out, _ = self.rnn(emb, hidden)
        logits = self.output(out.squeeze(0))
        return logits

    def generate(self, pb, start_token, max_len=10):
        """
        Generate a sentence from PB embedding.
        """
        hidden = self.pb_to_hidden(pb).unsqueeze(0).unsqueeze(0)
        token = torch.tensor([start_token])
        sentence = []

        for _ in range(max_len):
            emb = self.embedding(token).unsqueeze(0)
            out, hidden = self.rnn(emb, hidden)
            logits = self.output(out.squeeze(0))
            token = torch.argmax(logits, dim=-1)
            sentence.append(token.item())

        return sentence
