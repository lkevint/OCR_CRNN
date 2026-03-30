import torch
from torch import nn
from . import global_variables

class CRNN(nn.Module):
    def __init__(self, input_channels: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.block_2 = nn.Sequential(
            nn.Flatten(2, 3),
            nn.LSTM(512, 128, batch_first=True)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 128, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)                # (N, 64, 8, 32)
        x = torch.permute(x, (0, 3, 2, 1)) # (N, 32, 8, 64)
        x = self.block_2(x)[0]             # (N, 32, 128)
        x = self.classifier(x)             # (N, 46 * num_classes)
        x = torch.unflatten(x, 1, (global_variables.MAX_LABEL_LEN, len(global_variables.CHAR_LIST) + 1))
        x = torch.permute(x, (1, 0, 2))    # (T, N, C) for CTCLoss
        x = torch.log_softmax(x, dim=2)    # CTCLoss expects log-probs
        return x