import torch.nn as nn
from encoder import Encoder


class ClassificationModel(nn.Module):
    def __init__(
        self,
        seq_len=16,  # 4 for single digit
        d_model=64,
    ):
        super().__init__()
        self.encoder = Encoder(seq_len=seq_len, d_model=d_model)
        self.classifier = nn.Linear(d_model, 10)

    def forward(self, x):
        # Get encoder output
        encoder_output = self.encoder(x)

        # Use CLS token output (first token)
        cls_token = encoder_output[:, 0, :]

        # Project to classification space
        logits = self.classifier(cls_token)
        return logits
