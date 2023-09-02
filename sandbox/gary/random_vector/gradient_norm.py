#!/usr/bin/env python
# coding: utf-8

import os
import sys
from functools import partial
from pathlib import Path

import wandb
import click
from tqdm import tqdm
from absl import logging

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import torchopt
from torchmetrics import Accuracy
from torch.func import grad, grad_and_value, vmap, functional_call
from accelerate.utils import set_seed

from grablib import GraBSampler, BalanceType
from grablib.utils import EventTimer, pretty_time

# Change pwd to the project root directory
PROJECT_NAME = "GraB"
PROJECT_PATH = Path(__file__).resolve()
while PROJECT_PATH.name != PROJECT_NAME:
    PROJECT_PATH = PROJECT_PATH.parent
os.chdir(PROJECT_PATH)
sys.path.append(str(PROJECT_PATH))

DATASETS = ["mnist", "cifar10"]
MODELS = ["lr", "lenet"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(model, loss_fn, params, buffers, inputs, targets):
    logits = functional_call(model, (params, buffers), (inputs,))

    return loss_fn(logits, targets), logits


@torch.no_grad()
def train(
    train_loader,
    sampler,
    params,
    buffers,
    ft_compute_sample_grad_and_loss,
    metric,
    optimizer,
    opt_state,
    no_tqdm=False,
):
    norms = []

    running_loss = 0
    metric.reset()
    for _, (x, y) in tqdm(
        enumerate(train_loader), total=len(train_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)
        ft_per_sample_grads, (batch_loss, logits) = ft_compute_sample_grad_and_loss(
            params, buffers, x, y
        )

        b = x.shape[0]
        flattened_grads = torch.cat(
            [g.view(b, -1) for g in ft_per_sample_grads.values()], dim=1
        )

        norms.extend(torch.norm(flattened_grads, p=2, dim=1).tolist())

        sampler.step(ft_per_sample_grads)

        grads = {k: g.mean(dim=0) for k, g in ft_per_sample_grads.items()}
        updates, opt_state = optimizer.update(
            grads, opt_state, params=params
        )  # get updates
        params = torchopt.apply_updates(params, updates)  # update network parameters

        running_loss += batch_loss.mean()
        metric(logits, y)

    logging.debug(
        f"Gradient norms mean: {torch.mean(torch.tensor(norms)):.4f}"
        f" std: {torch.std(torch.tensor(norms)):.4f}"
        f" max: {torch.max(torch.tensor(norms)):.4f}"
        f" min: {torch.min(torch.tensor(norms)):.4f}"
    )

    return running_loss / len(train_loader)


# validation function
@torch.no_grad()
def validate(test_loader, model, loss_fn, metric, no_tqdm=False):
    running_loss = 0.0
    metric.reset()
    # look over the validation dataloader
    for _, (x, y) in tqdm(
        enumerate(test_loader), total=len(test_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        running_loss += loss.item()
        metric(outputs, y)
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
@click.option(
    "--balance",
    "balance_type",
    type=click.Choice([e.value for e in BalanceType], case_sensitive=False),
    default="mean",
)
@click.option("--prob", "prob_balance", is_flag=True)
@click.option("--prob-c", type=float, default=30)
@click.option("-d", "--depth", type=int, default=5)
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
    # grab specifics
    balance_type,
    prob_balance,
    prob_c,
    depth,
):
    name = balance_type
    config = {
        "balance_type": balance_type,
        "prob_balance": prob_balance,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
        "device": device,
    }

    if prob_balance:
        config["prob_c"] = prob_c
        name = f"{name}-prob-{prob_c}"

    if BalanceType(balance_type) in [
        BalanceType.RECURSIVE_BALANCE,
        BalanceType.RECURSIVE_PAIR_BALANCE,
    ]:
        config["depth"] = depth

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
    timer = EventTimer(device=device)

    if dataset == "mnist":
        raise NotImplementedError
    elif dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Loading the dataset and preprocessing
        train_dataset = datasets.CIFAR10(
            root="./data/external",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        test_dataset = datasets.CIFAR10(
            root="./data/external",
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
        from experiments.models import LeNet

        model = LeNet(in_dim, num_classes).to(device)
    else:
        raise ValueError
    # Transform everything to functional programming

    # Freeze all params in model
    for param in model.parameters():
        param.requires_grad = False

    # https://pytorch.org/docs/master/func.migrating.html#functorch-make-functional
    # https://pytorch.org/docs/stable/generated/torch.func.functional_call.html
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    if optimizer_name == "sgd":
        optimizer = torchopt.sgd(
            learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torchopt.adam(
            learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            use_accelerated_op=True,
        )
    else:
        raise ValueError

    opt_state = optimizer.init(params)  # init optimizer

    # Initiate sampler
    d = sum(p[1].numel() for p in model.named_parameters())
    logging.info(f"Number of parameters: d = {d:,}")
    sampler = GraBSampler(
        train_dataset,
        params,
        balance_type=balance_type,
        batch_size=batch_size,
        depth=depth,
        prob_balance=prob_balance,
        prob_balance_c=prob_c,
        device=device,
        timer=timer,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=sampler
    )
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size
    )

    # ft_compute_sample_grad = vmap(
    #     grad(partial(compute_loss, model, loss_fn), has_aux=True),
    #     in_dims=(None, None, 0, 0)
    # )  # the only argument of compute_loss is batched along the first axis

    ft_compute_sample_grad_and_loss = vmap(
        grad_and_value(partial(compute_loss, model, loss_fn), has_aux=True),
        in_dims=(None, None, 0, 0),
    )  # the only argument of compute_loss is batched along the first axis

    # loop over the dataloader multiple times
    for epoch in range(epochs):
        with timer(f"train"):
            # perform training (single loop over the train dataloader)
            train_loss = train(
                train_loader,
                sampler,
                params,
                buffers,
                ft_compute_sample_grad_and_loss,
                train_metric,
                optimizer,
                opt_state,
                no_tqdm=no_tqdm,
            )
        with timer("val"):
            # perform validation (single loop over the validation dataloader)
            val_loss = validate(
                test_loader, model, loss_fn, val_metric, no_tqdm=no_tqdm
            )

        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss :.3f} "
            f"train_acc: {train_metric.compute():.3f} | "
            f"val_loss: {val_loss :.3f} "
            f"val_acc: {val_metric.compute():.3f} | "
            f'train: {pretty_time(timer["train"][-1])} '
            f'val: {pretty_time(timer["val"][-1])}'
        )
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metric.compute(),
                "val_accuracy": val_metric.compute(),
                "train_time": timer["train"][-1],
                "val_time": timer["val"][-1],
            }
        )

    print(torch.cuda.memory_summary())
    print(timer.summary())
    peak_memory_allocated = (
        torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 / 1024
    )
    wandb.log(
        {
            "peak_gpu_mem": peak_memory_allocated,
            "total_train_time": sum(timer["train"]),
            "total_val_time": sum(timer["val"]),
        }
    )
    wandb.finish()


if __name__ == "__main__":
    cli()
