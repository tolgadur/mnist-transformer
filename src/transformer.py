import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from config import VOCAB


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        # Add output projection
        self.output_projection = nn.Linear(64, len(VOCAB))  # 64 is d_model

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x is input to encoder, i.e. image patches flattened
        # y is input to decoder, i.e. indices of labels
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(y, encoder_output)
        return self.output_projection(decoder_output)
