#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from timeit import default_timer as timer

import wandb
import click
from tqdm import tqdm
from absl import logging

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from torchmetrics import Accuracy
from accelerate.utils import set_seed

# Change pwd to the project root directory
PROJECT_NAME = "grab_exp"
PROJECT_PATH = Path(__file__).resolve()
while PROJECT_PATH.name != PROJECT_NAME:
    PROJECT_PATH = PROJECT_PATH.parent
os.chdir(PROJECT_PATH)

DATASETS = ["mnist", "cifar10"]
MODELS = ["lr", "lenet"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"


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
    metric.reset()
    for _, (x, y) in tqdm(
        enumerate(train_loader), total=len(train_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        metric(pred, y)

    return running_loss / len(train_loader)


# validation function
@torch.no_grad()
def validate(test_loader, model, loss_fn, metric, no_tqdm=False):
    model.eval()
    running_loss = 0.0
    metric.reset()
    # look over the validation dataloader
    for _, (x, y) in tqdm(
        enumerate(test_loader), total=len(test_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        running_loss += loss.item()
        metric(pred, y)
    return running_loss / len(test_loader)


@click.command()
@click.argument("dataset", type=click.Choice(DATASETS), default=DATASETS[-1])
@click.argument("model_name", type=click.Choice(MODELS), default=MODELS[-1])
@click.option("-lr", "--learning-rate", type=float, default=1e-3)
@click.option("-wd", "--weight-decay", type=float, default=1e-2)
@click.option("-b", "--batch-size", type=int, default=16)
@click.option("-e", "--epochs", type=int, default=10)
@click.option(
    "-opt",
    "--optimizer",
    "optimizer_name",
    type=click.Choice(["sgd", "adam"]),
    default="sgd",
)
@click.option("-m", "--momentum", type=float, default=0.9)
@click.option("-b1", "--beta1", type=float, default=0.9)
@click.option("-b2", "--beta2", type=float, default=0.999)
@click.option("-s", "--seed", type=int, default=42)
@click.option(
    "-g",
    "logging_level",
    type=click.Choice(["fatal", "error", "warning", "info", "debug"]),
    default="warning",
)
@click.option("--no-tqdm", is_flag=True)
@click.option("--wandb", "sync_wandb", is_flag=True)
def cli(
    # task specifics
    dataset,
    # task,  # auto inferred for now
    model_name,
    # training specifics
    learning_rate,
    weight_decay,
    batch_size,
    epochs,
    optimizer_name,
    momentum,
    beta1,
    beta2,
    seed,
    # helper parameters
    logging_level,
    no_tqdm,
    sync_wandb,
):
    name = "native_pytorch"
    config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
        "device": device,
    }

    # Initialize a wandb run
    wandb.init(
        project=f"grab-{dataset}",
        entity="grab",
        name=name,
        mode="online" if sync_wandb else "offline",
        config=config,
    )

    set_seed(seed)
    logging.set_verbosity(
        {
            "fatal": logging.FATAL,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }[logging_level]
    )

    if dataset == "mnist":
        raise NotImplementedError
    elif dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Loading the dataset and preprocessing
        train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        in_dim, num_classes = 3, 10

        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError

    train_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    if model_name == "lr":
        raise NotImplementedError
    elif model_name == "lenet":
        from models import LeNet

        model = LeNet(in_dim, num_classes).to(device)
    else:
        raise ValueError

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "adam":
        raise NotImplementedError
    else:
        raise ValueError

    # Initiate sampler
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size
    )

    train_time, val_time = 0, 0

    # loop over the dataloader multiple times
    for epoch in range(epochs):
        _start_time = timer()
        # perform training (single loop over the train dataloader)
        train_loss = train(
            train_loader, model, loss_fn, train_metric, optimizer, no_tqdm=no_tqdm
        )
        _train_time = timer()
        # perform validation (single loop over the validation dataloader)
        val_loss = validate(test_loader, model, loss_fn, val_metric, no_tqdm=no_tqdm)
        _end_time = timer()

        elapsed_train = _train_time - _start_time
        elapsed_val = _end_time - _train_time

        train_time += elapsed_train
        val_time += elapsed_val

        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss :.3f} "
            f"train_acc: {train_metric.compute():.3f} | "
            f"val_loss: {val_loss :.3f} "
            f"val_acc: {val_metric.compute():.3f} | "
            f"train: {elapsed_train:.2f}s "
            f"val: {elapsed_val:.2f}s"
        )
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metric.compute(),
                "val_accuracy": val_metric.compute(),
                "train_time": elapsed_train,
                "val_time": elapsed_val,
            }
        )

    print(torch.cuda.memory_summary())
    peak_memory_allocated = (
        torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 / 1024
    )
    wandb.log(
        {
            "peak_gpu_mem": peak_memory_allocated,
            "total_train_time": train_time,
            "total_val_time": val_time,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    cli()
