import torch
import torch.nn as nn


# todo: residual connection, multi-head attention
class EncoderBlock(nn.Module):
    def __init__(self, input_dim=196, dff=1024, seq_len=16, d_model=64, dropout=0.1):
        super().__init__()

        self.embedding = nn.linear(input_dim, d_model)
        self.positional_encoding = nn.parameter.Parameter(nn.randn(seq_len, d_model))

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding

        # attention-head
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        A = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(self.d_model))
        A = torch.softmax(A, dim=-1)
        A = torch.matmul(A, V)
        A = self.dropout(A)
        A = self.layer_norm(A)

        # residual connection
        x = x + A

        # feed-forward with residual connection
        return x + self.linear(x)
