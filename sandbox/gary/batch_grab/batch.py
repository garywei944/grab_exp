#!/usr/bin/env python
# coding: utf-8

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
from pprint import pprint

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import HfArgumentParser, TrainingArguments, set_seed

import torchopt
from torch.func import grad, grad_and_value, vmap, functional_call

from grablib import GraBSampler, BalanceType
from grablib.utils import EventTimer, pretty_time

from experiments.cv.models import LeNet


device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = b = 16
epochs = 100


def train(
    train_loader,
    model,
    loss_fn,
    metric,
    optimizer,
    no_tqdm=False,
):
    model.train()
    running_loss = 0
    num_examples = 0
    for _, (x, y) in tqdm(
        enumerate(train_loader), total=len(train_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_examples += x.shape[0]
        running_loss += loss.item() * x.shape[0]
        metric.add_batch(predictions=pred.argmax(dim=-1), references=y)

        print(grab_beta.grad)
        grab_beta.grad.zero_()

    return running_loss / num_examples


@torch.no_grad()
def validate(model, loss_fn, metric, test_loader, no_tqdm=False):
    model.eval()
    running_loss = 0
    num_examples = 0
    for _, (x, y) in tqdm(
        enumerate(test_loader), total=len(test_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        num_examples += x.shape[0]
        running_loss += loss.item() * x.shape[0]
        metric.add_batch(predictions=pred.argmax(dim=-1), references=y)

    return running_loss / num_examples


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.49139968, 0.48215841, 0.44653091],
            std=[0.24703223, 0.24348513, 0.26158784],
        ),
    ]
)

# Loading the dataset and preprocessing
train_dataset = datasets.CIFAR10(
    root="data/external", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="data/external", train=False, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
train_eval_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

in_dim, num_classes = 3, 10
loss_fn = nn.CrossEntropyLoss().to(device)

set_seed(42)


model = LeNet(in_dim, num_classes).to(device)
grab_beta = torch.zeros(b, device=device, requires_grad=True)

d = sum(p[1].numel() for p in model.named_parameters())
logging.info(f"Number of training examples: n = {len(train_dataset):,}")
logging.info(f"Number of parameters: d = {d:,}")


# GraB hook
def forward_hood(model, input):
    assert len(input) == 1
    w = input[0]
    print(id(w))
    return w + torch.einsum("b,b...->b...", grab_beta, w)


# Check that all layers are only hooked once
total = 0

acc = {}
for name, module in model.named_modules():
    if len(list(module.children())) > 0:
        continue
    s = sum(p.numel() for p in module.parameters())
    if s > 0:
        total += s
        module.register_forward_pre_hook(forward_hood)

print(total)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, weight_decay=1e-4)
# Create metrics
train_metric = evaluate.load("accuracy")
train_eval_metric = evaluate.load("accuracy")
val_metric = evaluate.load("accuracy")

for epoch in range(epochs):
    print(f"Epoch {epoch}")
    train_loss = train(train_loader, model, loss_fn, train_metric, optimizer)
    print(f"Train loss: {train_loss:.4f}")
    train_eval_loss = validate(model, loss_fn, train_eval_metric, train_eval_loader)
    train_eval_acc = train_eval_metric.compute()["accuracy"]
    print(f"Train eval loss: {train_eval_loss:.4f} acc: {train_eval_acc:.4f}")

    val_loss = validate(model, loss_fn, val_metric, test_loader)
    val_acc = val_metric.compute()["accuracy"]
    print(f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

    print(grab_beta)
    print(grab_beta.grad.shape)
