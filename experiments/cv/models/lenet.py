from torch import nn


# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network
class LeNet(nn.Module):
    def __init__(self, in_dim=3, out_dim=10):
        super(LeNet, self).__init__()
        # # 1 input image channel, 6 output channels, 5x5 square convolution
        # # kernel
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, out_dim))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 120)
        x = self.fc(x)
        return x
