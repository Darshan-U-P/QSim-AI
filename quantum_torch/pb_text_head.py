import torch
import torch.nn as nn


class PBTextHead(nn.Module):
    """
    PB → Language head
    Converts PB embeddings into short sentences using a GRU decoder.

    Architecture:
        PB embedding → hidden state
        <bos> token → embedding → GRU → token logits
    """

    def __init__(self, pb_dim, vocab, embed_dim=32, hidden_dim=64):
        super().__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.token_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_token = {i: w for i, w in enumerate(vocab)}

        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        self.pb_to_hidden = nn.Linear(pb_dim, hidden_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, pb, input_ids):
        """
        pb        : (PB_DIM,)
        input_ids : (1, seq_len)
        """
        hidden = self.pb_to_hidden(pb)             # (hidden_dim,)
        hidden = hidden.unsqueeze(0).unsqueeze(0)  # (1,1,hidden_dim)

        embed = self.embed(input_ids)              # (1, seq_len, embed_dim)
        out, _ = self.gru(embed, hidden)           # (1, seq_len, hidden_dim)
        logits = self.fc_out(out)                  # (1, seq_len, vocab)

        return logits

    def generate(self, pb, max_len=6):
        """
        Greedy sentence generation from PB embedding.
        pb must be shape: (PB_DIM,)
        """
        device = pb.device

        # Force PB to be 1D
        if pb.dim() > 1:
            pb = pb.view(-1)

        bos_id = self.token_to_id["<bos>"]
        eos_id = self.token_to_id["<eos>"]

        # Initial input token <bos>
        input_id = torch.tensor([[bos_id]], device=device)  # (1,1)

        # Initial hidden state from PB
        hidden = self.pb_to_hidden(pb)        # (hidden_dim,)
        hidden = hidden.unsqueeze(0).unsqueeze(0)  # (1,1,hidden_dim)

        generated = []

        for _ in range(max_len):
            embed = self.embed(input_id)           # (1,1,embed_dim)

            # GRU step
            out, hidden = self.gru(embed, hidden) # hidden stays (1,1,hidden_dim)

            logits = self.fc_out(out[:, -1, :])    # (1,vocab)
            token_id = torch.argmax(logits, dim=-1).item()

            if token_id == eos_id:
                break

            token = self.id_to_token[token_id]
            generated.append(token)

            input_id = torch.tensor([[token_id]], device=device)

        return " ".join(generated)
