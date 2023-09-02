from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=100, out_dim=10):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.linear(x)
