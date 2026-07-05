import torch
import torch.nn as nn


class FolkLSTM(nn.Module):

    def __init__(self):

        super(FolkLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=13,      # 13 MFCC coefficients
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(
            128,
            13
        )

    def forward(self, x):

        # x shape: (batch_size, 13, 431)

        x = x.permute(0, 2, 1)

        # Now shape becomes:
        # (batch_size, 431, 13)

        output, _ = self.lstm(x)

        output = self.fc(output)

        return output