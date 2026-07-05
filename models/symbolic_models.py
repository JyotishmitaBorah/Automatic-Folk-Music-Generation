import torch
import torch.nn as nn


class FolkLSTMv2(nn.Module):
    """
    Symbolic LSTM for next-token prediction.

    Input:
        token ids

    Output:
        probability of next token
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x, hidden=None):

        x = self.embedding(x)

        output, hidden = self.lstm(x, hidden)

        output = self.dropout(output)

        logits = self.fc(output)

        return logits, hidden