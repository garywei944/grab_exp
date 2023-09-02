from torchvision import datasets

cifar_trainset = datasets.CIFAR10(root="~/projects/GraB-lib/data/external", train=True, download=True)
data = cifar_trainset.data / 255  # data is numpy array

mean = data.mean(axis=(0, 1, 2))
std = data.std(axis=(0, 1, 2))
print(
    f"Mean : {mean}   STD: {std}"
)  # Mean : [0.491 0.482 0.446]   STD: [0.247 0.243 0.261]
