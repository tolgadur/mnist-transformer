import torch.nn as nn
from encoder import Encoder


class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes=9999,  # 10 for single digit
        seq_len=16,  # 4 for single digit
        d_model=64,
    ):
        super().__init__()
        self.encoder = Encoder(seq_len=seq_len, d_model=d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Get encoder output
        encoder_output = self.encoder(x)
        # Use CLS token output (first token)
        cls_token = encoder_output[:, 0, :]
        # Project to classification space
        logits = self.classifier(cls_token)
        return logits
