from torchvision import datasets

# cifar_trainset = datasets.CIFAR10(
#     root="~/projects/grab_exp/data/external", train=True, download=True
# )
# data = cifar_trainset.data / 255  # data is numpy array

# MNIST
data = (
    datasets.FashionMNIST(
        root="~/projects/grab_exp/data/external", download=True
    ).data
    / 255
)

mean = data.mean(axis=(0, 1, 2))
std = data.std(axis=(0, 1, 2))
print(
    f"Mean : {mean}   STD: {std}"
)  # Mean : [0.491 0.482 0.446]   STD: [0.247 0.243 0.261]
