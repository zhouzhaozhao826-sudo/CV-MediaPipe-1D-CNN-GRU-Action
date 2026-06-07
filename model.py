from __future__ import annotations

import torch
from torch import nn


class SquatCNNGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_channels: int = 64,
        gru_hidden_size: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.temporal_model = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        _, hidden = self.temporal_model(x)
        logits = self.classifier(hidden[-1])
        return logits
