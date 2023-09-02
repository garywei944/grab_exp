import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, inp):
        return self.fc2(self.fc1(inp))


# def _backward_hook(module, grad_input, grad_output):
#     print(module)
#     print(type(grad_input))
#     print(len(grad_input))
#     print(type(grad_output))
#     print(len(grad_output))
#     [_grads.append(g) for g in grad_input]
#     [_grads.append(g) for g in grad_output]
#     # print(grad_input[0].shape)
#     # # print(grad_input[1].shape)
#     # print(grad_output[0].shape)
#     print()\

F_in = []
D_out = []


def _forward_hook(module, input, output):
    print(input)
    print(input[0].shape)
    # print(output)
    # print(output[0].shape)
    F_in.append(input[0].detach().clone())
    print()


def _backward_hook(module, grad_input, grad_output):
    # model, grad_input, grad_output = args
    # # print(list(model.parameters()))
    # # print(grad_input)
    # # print([p * grad_input[0] for p in model.parameters()])
    print(grad_output)
    # print([p for p in model.parameters()])
    print(grad_output[0].shape)
    D_out.append(grad_output[0].detach().clone())

    print([p.grad for p in module.parameters()])
    print()

def _pre_backward_hook(module, grad_input):

    print([p.grad for p in module.parameters()])
    print()


model = Model()
# d = sum(p.numel() for p in model.parameters())

# grad_hook = torch.zeros(d)

[m.register_forward_hook(_forward_hook) for m in model.children()]
[m.register_full_backward_hook(_backward_hook) for m in model.children()]
# [m.register_full_backward_pre_hook(_pre_backward_hook) for m in model.children()]
# print([m for m in model.named_children()])
out = model(torch.arange(30).reshape(3, 10).float())
out.mean().backward(retain_graph=True)

grads = torch.cat([p.grad.flatten() for p in model.parameters()])
print(grads)
print(grads.shape)

_grads = []
for f_in, d_out in zip(F_in, D_out[::-1]):
    _grads.append(torch.einsum("bi,bj->ij", d_out, f_in).flatten())
    _grads.append(d_out.sum(dim=0).flatten())
    print(f_in.shape, d_out.shape)
_grads = torch.cat(_grads)
print(_grads)
print(_grads.shape)

# Gary: In short, to construct per layer gradient via hook, we need to store all per
# layer input and know prior knowledge about layer.

print(torch.allclose(grads, _grads))
