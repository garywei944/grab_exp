import torch
import torch.nn as nn

from torchinfo import summary
from experiments.cv.models import MinLeNet, LeNet

net = MinLeNet(3, 10).cuda()
# net = LeNet(3, 10).cuda()

x = torch.rand((3, 32, 32)).cuda()

summary(net, input_size=(1, 3, 32, 32), verbose=2)

print(net(x))
