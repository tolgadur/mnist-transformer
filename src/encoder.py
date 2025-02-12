import torch.nn as nn
from attention import Attention


class Encoder(nn.Module):
    def __init__(self, input_dim=196, dff=1024, seq_len=16, d_model=64, dropout=0.1):
        super().__init__()

        self.embedding = nn.linear(input_dim, d_model)
        self.positional_encoding = nn.parameter.Parameter(nn.randn(seq_len, d_model))

        self.encoder_block = nn.Sequential(
            EncoderBlock(dff, d_model, dropout),
            EncoderBlock(dff, d_model, dropout),
            EncoderBlock(dff, d_model, dropout),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding

        x = self.encoder_block(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, dff=1024, d_model=64, dropout=0.1):
        super().__init__()

        self.attention = Attention(d_model, dropout)
        self.linear = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # multi-head attention with residual connection
        x = self.attention(x)

        # feed-forward with residual connection
        return x + self.linear(x)
