import torch.nn as nn
import separableconv.nn as sconv


class MinNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
            nn.InstanceNorm2d(8),
            nn.MaxPool2d(2),
            # nn.Dropout(0.1),
        )
        self.conv2 = sconv.SeparableConv2d(
            8, 26, 3, normalization_dw="in", normalization_pw="in", bias=True
        )
        self.conv3 = sconv.SeparableConv2d(
            26,
            26,
            3,
            padding=1,
            normalization_dw="in",
            normalization_pw="in",
            bias=True,
        )
        self.pool = nn.MaxPool2d(11)

        self.cls = nn.Sequential(
            nn.Linear(26, 16),
            # nn.Dropout(0.1),
            nn.Linear(16, 10),
            nn.Softmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(-1, 26).squeeze()
        return self.cls(x)
