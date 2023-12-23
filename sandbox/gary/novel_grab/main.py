import numpy as np
import torch
from torch import Tensor
import random


def gen_random_vectors(n, d, dtype=torch.float32, device="cuda"):
    vec = torch.normal(0, 1, (n, d), dtype=dtype, device=device)
    vec /= vec.norm(dim=1).max()  # scale to unit norm
    vec -= vec.mean(dim=0)  # center at origin
    return vec


def herding(vec: Tensor, p: int | str = 2):
    if p == 2:
        return torch.cumsum(vec, dim=0).norm(dim=0).max().item()
    else:
        return torch.cumsum(vec, dim=0).abs().max().item()


def grab(vec: Tensor, noise: bool = False) -> Tensor:
    n, d = vec.shape

    acc = torch.zeros(d, dtype=vec.dtype, device=vec.device)
    orders = torch.zeros(n, dtype=torch.int64)

    z = torch.randn(d, dtype=vec.dtype, device=vec.device) * 0.05 if noise else 0

    left, right = 0, n - 1
    for i in range(n):
        if torch.inner(vec[i], acc) < 0:
            # print(f'stat: {torch.inner(vec[i], acc)}')
            # if random.random() < 0.5 - torch.inner(
            #     vec[i] / vec[i].norm(), acc / acc.norm()
            # )*8:
            acc += vec[i] + z
            orders[left] = i
            left += 1
        else:
            acc -= vec[i] + z
            orders[right] = i
            right -= 1

    assert left == right + 1

    return orders


def main():
    n = 1024
    d = 128

    V = gen_random_vectors(n, d)

    V_copy = V.clone()

    print("Random Reshufling")
    for _ in range(10):
        print(herding(V[torch.randperm(n, dtype=torch.int64)]))

    print("-" * 20)
    print("GraB")

    for _ in range(10):
        # print("-" * 20)
        # print(f"Iteration {i}")
        print(herding(V))
        orders = grab(V)
        # print(herding(V[torch.randperm(n, dtype=torch.int64)]))
        V = V[orders]

    print(herding(V))

    V.copy_(V_copy)

    print("-" * 20)
    print("GraB with Noise adding to accumulator")

    for _ in range(10):
        # print("-" * 20)
        # print(f"Iteration {i}")
        print(herding(V))
        orders = grab(V, noise=True)
        # print(herding(V[torch.randperm(n, dtype=torch.int64)]))
        V = V[orders]

    print(herding(V))


if __name__ == "__main__":
    main()
