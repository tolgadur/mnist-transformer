import torch
import torch.nn as nn
from attention import Attention


class Encoder(nn.Module):
    def __init__(self, input_dim=196, dff=256, seq_len=16, d_model=64, dropout=0.1):
        super().__init__()

        # Learnable class token
        self.cls_token = nn.parameter.Parameter(torch.randn(1, d_model))

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.parameter.Parameter(
            torch.randn(seq_len + 1, d_model)
        )

        self.encoder_block = nn.Sequential(
            EncoderBlock(dff, d_model, dropout),
            EncoderBlock(dff, d_model, dropout),
            EncoderBlock(dff, d_model, dropout),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.embedding(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.positional_encoding
        x = self.encoder_block(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, dff=256, d_model=64, dropout=0.1):
        super().__init__()

        self.attention = Attention(d_model, dropout)
        self.linear = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # multi-head attention with residual connection
        x = self.attention(x)

        # feed-forward with residual connection
        return x + self.linear(x)
