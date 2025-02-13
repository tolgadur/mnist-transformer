import torch
import torch.nn as nn
from attention import Attention, CrossAttention
from mlp import MultiLayerPerceptron


class Decoder(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.decoder_layers = nn.Sequential(
            *[DecoderLayer(d_model, dropout, heads) for _ in range(6)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.decoder_layers(x, y)


class DecoderLayer(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.self_attn = Attention(d_model, dropout, heads)
        self.cross_attn = CrossAttention(d_model, dropout, heads)
        self.mlp = MultiLayerPerceptron(d_model, dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x is input to the decoder, y is input to the encoder
        # dim of x: batch_size, seq_len_x, d_model
        # dim of y: batch_size, seq_len_y, d_model

        x = self.self_attn(x)
        x = self.cross_attn(x, y)
        x = self.mlp(x)
        return x
