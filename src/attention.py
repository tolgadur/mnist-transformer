import torch
import torch.nn as nn


# todo: add masking
class Attention(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.d_k = d_model // heads
        self.d_v = self.d_k
        self.heads = heads

        self.Qs = [nn.Linear(self.d_k, self.d_k) for _ in range(heads)]
        self.Ks = [nn.Linear(self.d_k, self.d_k) for _ in range(heads)]
        self.Vs = [nn.Linear(self.d_v, self.d_v) for _ in range(heads)]

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        As = []
        for i in range(self.heads):
            Q = self.Qs[i](x)
            K = self.Ks[i](x)
            V = self.Vs[i](x)

            A = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(self.d_k))
            A = torch.softmax(A, dim=-1)  # todo: why dim=-1?
            A = torch.matmul(A, V)

            As.append(A)

        A = torch.cat(As, dim=-1)
        A = self.dropout(A)
        A = self.layer_norm(A)

        return x + A
