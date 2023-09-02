import torch
import separableconv.nn as nn
import torch.nn.functional as F


class MinLeNet(nn.Module):
    def __init__(self, in_dim=3, out_dim=10):
        super(MinLeNet, self).__init__()
        # self.conv1 = nn.SeparableConv2d(
        #     in_dim, 6, 5,
        #     normalization_dw='in',
        #     normalization_pw='in',
        # )
        self.conv2 = nn.SeparableConv2d(
            6,
            16,
            5,
            normalization_dw="in",
            normalization_pw="in",
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, 6, 5),
            nn.ReLU(),
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(6, 16, 5),
        #     nn.ReLU(),
        # )
        # self.conv3 = nn.SeparableConv2d(
        #     16, 120, 5,
        #     normalization_dw='',
        #     normalization_pw='',
        # )
        self.conv3 = nn.Conv2d(16, 64, 5)

        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, out_dim))

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2, 2))
        x = F.max_pool2d(self.conv2(x), (2, 2))
        x = self.conv3(x)
        x = x.view(-1, 64).squeeze()

        x = self.fc(x)
        return x
