import torch
import torch.nn as nn
from attention import Attention, CrossAttention
from mlp import MultiLayerPerceptron
from config import VOCAB


class Decoder(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4, max_seq_len=100):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=len(VOCAB), embedding_dim=d_model)
        self.positional_encoding = nn.parameter.Parameter(
            torch.randn(max_seq_len, d_model)
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, dropout, heads) for _ in range(6)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # dim: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = x + self.positional_encoding[: x.size(1)]
        for layer in self.layers:
            x = layer(x, y)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.self_attn = Attention(d_model, dropout, heads, apply_mask=True)
        self.cross_attn = CrossAttention(d_model, dropout, heads)
        self.mlp = MultiLayerPerceptron(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x is input to the decoder, y is input to the encoder
        # dim of x: batch_size, seq_len_x, d_model
        # dim of y: batch_size, seq_len_y, d_model
        x = self.self_attn(x)
        x = self.cross_attn(x, y)
        x = self.mlp(x)
        return x
