import torch


class TestFunc(torch.nn.Module):
    def __init__(self):
        super(TestFunc, self).__init__()

        self.alpha = torch.nn.Parameter(torch.tensor([100.0]), requires_grad=True)

    def forward(self, input: torch.Tensor):
        x_1 = torch.roll(input, +1)
        return torch.sum(self.alpha * (x_1 - input**2) ** 2 + (1 - input) ** 2, 0)


input = torch.rand(2, requires_grad=True) * 5

model = TestFunc()

model.register_parameter(
    "gamma", torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
)

model.alpha = torch.nn.Parameter(torch.exp(model.gamma), requires_grad=True)

output2 = model(input)

output2.backward()

print(model.alpha)
print(model.alpha.grad)
print(model.gamma)
print(model.gamma.grad)
