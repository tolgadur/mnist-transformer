import torch
import torch.nn as nn
from attention import Attention
from mlp import MultiLayerPerceptron


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=196,
        dff=256,
        seq_len=16,
        d_model=64,
        dropout=0.1,
        heads=4,
        cls_token=False,
    ):
        super().__init__()

        # Store cls_token flag
        self.use_cls_token = cls_token

        # Learnable class token - only create if needed
        if self.use_cls_token:
            self.cls_token = nn.parameter.Parameter(torch.randn(1, 1, d_model))
            # Adjust positional encoding length for CLS token
            self.positional_encoding = nn.parameter.Parameter(
                torch.randn(seq_len + 1, d_model)
            )
        else:
            self.cls_token = None
            self.positional_encoding = nn.parameter.Parameter(
                torch.randn(seq_len, d_model)
            )

        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layers = nn.Sequential(
            *[EncoderBlock(dff, d_model, dropout, heads) for _ in range(6)]
        )

    def forward(self, x):
        x = self.embedding(x)

        if self.use_cls_token:
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.positional_encoding
        x = self.encoder_layers(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, dff=256, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.attention = Attention(d_model, dropout, heads)
        self.mlp = MultiLayerPerceptron(d_model, dff, dropout)

    def forward(self, x):
        # multi-head attention with residual connection
        x = self.attention(x)

        # feed-forward with residual connection
        return x + self.mlp(x)
