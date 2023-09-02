from torch import nn

import os
import sys
from functools import partial, reduce
from pathlib import Path
from dataclasses import dataclass, field

import evaluate
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from absl import logging

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import (
    HfArgumentParser, TrainingArguments, set_seed
)

import torchopt
from torch.func import (
    grad, grad_and_value, vmap, functional_call
)

from grablib import GraBSampler, BalanceType
from grablib.utils import EventTimer, pretty_time


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
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 120).squeeze()
        x = self.fc(x)
        return x


def main():
    device = 'cuda'
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

    # Loading the dataset and preprocessing
    train_dataset = datasets.CIFAR10(
        root='data/external',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='data/external',
        train=False,
        download=True,
        transform=transform
    )

    in_dim, num_classes = 3, 10

    loss_fn = nn.CrossEntropyLoss().to(device)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # sampler=sampler,
        persistent_workers=False,
        num_workers=1,
        pin_memory=True
    )
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        persistent_workers=False,
        num_workers=1,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        persistent_workers=False,
        num_workers=1,
        pin_memory=True
    )

    nets = []
    for seed in range(2):
        set_seed(seed)
        nets.append(LeNet().to(device))

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        for net in nets:
            out = net(x)
            out.backward()
            grad = [p.grad.view(-1) for p in net.parameters()]
            grads = torch.cat(grad)
            print(grads.shape)
        break


if __name__ == '__main__':
    main()
