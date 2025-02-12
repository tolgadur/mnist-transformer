import torch
import torch.nn as nn


# todo: add masking
class Attention(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.d_k = d_model // heads
        self.d_v = self.d_k
        self.heads = heads
        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        self.Qs = nn.ModuleList([nn.Linear(self.d_k, self.d_k) for _ in range(heads)])
        self.Ks = nn.ModuleList([nn.Linear(self.d_k, self.d_k) for _ in range(heads)])
        self.Vs = nn.ModuleList([nn.Linear(self.d_v, self.d_v) for _ in range(heads)])

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):

        As = []
        head_chunks = torch.chunk(x, self.heads, dim=-1)
        for i in range(self.heads):
            Q = self.Qs[i](head_chunks[i])  # dim: batch_size, seq_len, d_k
            K = self.Ks[i](head_chunks[i])  # dim: batch_size, seq_len, d_k
            V = self.Vs[i](head_chunks[i])  # dim: batch_size, seq_len, d_v

            # only transpose last two layers of tensor, not the batch size
            A = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            A = torch.softmax(A, dim=-1)  # todo: why dim=-1?
            A = torch.matmul(A, V)  # dim: batch_size, seq_len, d_v

            As.append(A)

        A = torch.cat(As, dim=-1)
        A = self.dropout(A)
        A = A + x  # residual connection
        A = self.layer_norm(A)

        return A
