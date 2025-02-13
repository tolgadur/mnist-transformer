import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x is input to encoder, i.e. image patches flattened
        # y is input to decoder, i.e. indices of labels
        return self.decoder(y, self.encoder(x))
