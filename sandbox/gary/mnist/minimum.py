#!/usr/bin/env python
# coding: utf-8

import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from absl import logging

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from transformers import HfArgumentParser, TrainingArguments
import evaluate

import torchinfo

from model import Net


def main():
    parser = HfArgumentParser((TrainingArguments,))

    (args,) = parser.parse_args_into_dataclasses()

    # Dataset

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Lambda(lambda x: x.view(-1))
        ]
    )
    train_dataset = datasets.MNIST(
        root="data/external", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data/external", train=False, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False
    )

    loss_fn = nn.CrossEntropyLoss().to(args.device)

    # Model
    model = Net().to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    process_bar = tqdm(range(int(args.num_train_epochs)))

    for epoch in range(int(args.num_train_epochs)):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.logging_steps == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
        )


if __name__ == "__main__":
    main()
