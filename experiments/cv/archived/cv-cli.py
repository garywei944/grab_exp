#!/usr/bin/env python
# coding: utf-8

import os
import sys
from functools import partial
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import wandb
import click
from tqdm import tqdm
from absl import logging

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import torchopt
from torch.func import grad, grad_and_value, vmap, functional_call
from accelerate.utils import set_seed

from grablib import GraBSampler, BalanceType
from grablib.utils import EventTimer, pretty_time

# Change pwd to the project root directory
PROJECT_NAME = "grab_exp"
PROJECT_PATH = Path(__file__).resolve()
while PROJECT_PATH.name != PROJECT_NAME:
    PROJECT_PATH = PROJECT_PATH.parent
os.chdir(PROJECT_PATH)
sys.path.insert(0, str(PROJECT_PATH))

DATASETS = ["mnist", "cifar10"]
MODELS = ["lr", "lenet", "minnet"]

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
    optimizer,
    opt_state,
    metric: evaluate.EvaluationModule,
    no_tqdm: bool = False,
):
    running_loss = 0
    for _, (x, y) in tqdm(
        enumerate(train_loader), total=len(train_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)
        ft_per_sample_grads, (batch_loss, logits) = ft_compute_sample_grad_and_loss(
            params, buffers, x, y
        )

        sampler.step(ft_per_sample_grads)

        grads = {k: g.mean(dim=0) for k, g in ft_per_sample_grads.items()}
        updates, opt_state = optimizer.update(
            grads, opt_state, params=params
        )  # get updates
        params = torchopt.apply_updates(params, updates)  # update network parameters

        running_loss += batch_loss.mean()
        metric.add_batch(predictions=logits.argmax(dim=-1), references=y)

    return running_loss / len(train_loader)


# validation function
@torch.no_grad()
def validate(
    test_loader, model, loss_fn, metric: evaluate.EvaluationModule, no_tqdm=False
):
    running_loss = 0.0
    # look over the validation dataloader
    for _, (x, y) in tqdm(
        enumerate(test_loader), total=len(test_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        running_loss += loss.item()
        metric.add_batch(predictions=outputs.argmax(dim=-1), references=y)
    return running_loss / len(test_loader)


@click.command()
@click.argument("dataset", type=click.Choice(DATASETS), default=DATASETS[-1])
@click.argument("model_name", type=click.Choice(MODELS), default=MODELS[-1])
@click.option("-o", "--output", "output_path", type=click.Path(), default="checkpoints")
@click.option("--save", "save_checkpoints", is_flag=True)
@click.option("-nw", "--num-workers", type=int, default=0)
@click.option(
    "--dtype", type=click.Choice(["float32", "float16", "bfloat16"]), default="float32"
)
@click.option("--cuda-herding", is_flag=True)
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
@click.option("--no-rr", "no_random_first_epoch", is_flag=True)
@click.option("--norm", "normalize", is_flag=True)
@click.option("--pi", "random_projection", is_flag=True)
@click.option("--eps", "pi_eps", type=float, default=0.1)
@click.option("--prob", "prob_balance", is_flag=True)
@click.option("--prob-c", type=float, default=30)
@click.option("-d", "--depth", type=int, default=5)
@click.option("--ema-decay", type=float, default=0.1)
def cli(
    # task specifics
    dataset,
    # task,  # auto inferred for now
    model_name,
    output_path,
    save_checkpoints,
    num_workers,
    dtype,
    cuda_herding,
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
    no_random_first_epoch,
    normalize,
    random_projection,
    pi_eps,
    prob_balance,
    prob_c,
    depth,
    ema_decay,
):
    # check if it is recursive
    is_recursive = BalanceType(balance_type) in [
        BalanceType.RECURSIVE_BALANCE,
        BalanceType.RECURSIVE_PAIR_BALANCE,
    ]

    checkpoint_path = Path(output_path) / f"{dataset}-{model_name}" / balance_type
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Unique experiment name for checkpoints
    exp_id = (
        f"{dataset}_{model_name}_{balance_type}"
        f'{f"_norm" if normalize else ""}'
        f'{f"_pi_{pi_eps}" if random_projection else ""}'
        f'{f"_prob_{prob_c:.1f}" if prob_balance else ""}'
        f'{f"_depth_{depth}" if is_recursive else ""}'
        f"_{optimizer_name}_lr_{learning_rate}_wd_{weight_decay}"
        f"_b_{batch_size}_seed_{seed}"
        f'{f"_no_rr" if no_random_first_epoch else ""}'
        f"ema_{ema_decay}"
    )

    name = balance_type
    config = {
        "balance_type": balance_type,
        "random_first_epoch": not no_random_first_epoch,
        "normalize": normalize,
        "random_projection": random_projection,
        "pi_eps": pi_eps,
        "prob_balance": prob_balance,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
        "dtype": dtype,
        "device": device,
        "ema_decay": ema_decay,
    }

    if normalize:
        name += "-norm"
    if random_projection:
        name += "-pi"
    if prob_balance:
        name += "-prob"

    if prob_balance:
        config["prob_c"] = prob_c
        name += f"-prob-{prob_c}"

    if is_recursive:
        config["depth"] = depth

    # Initialize a wandb run
    wandb.init(
        project=f"grab-{dataset}",
        entity="grab",
        name=name,
        mode="online" if sync_wandb else "offline",
        config=config,
    )

    # Set up seed, logging, and timer
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

    # Set dtype, only used for the sampler
    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]

    logging.info(f"Name: {name}")
    logging.info(f"Experiment ID: {exp_id}")
    logging.info(config)

    # Load the dataset
    if dataset == "mnist":
        if model_name == "minnet":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.view(-1)),
                ]
            )
        train_dataset = datasets.MNIST(
            root="data/external", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="data/external", train=False, transform=transform
        )

        in_dim, num_classes = 784, 10

        loss_fn = nn.CrossEntropyLoss().to(device)
    elif dataset == "cifar10":
        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Loading the dataset and preprocessing
        train_dataset = datasets.CIFAR10(
            root="data/external",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transform]),
        )

        test_dataset = datasets.CIFAR10(
            root="data/external",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transform]),
        )

        in_dim, num_classes = 3, 10

        loss_fn = nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError

    # Create metrics
    train_metric = evaluate.load("accuracy")
    train_eval_metric = evaluate.load("accuracy")
    val_metric = evaluate.load("accuracy")

    # Load the model
    if model_name == "lr":
        assert dataset == "mnist"

        model = nn.Linear(in_dim, num_classes).to(device)
    elif model_name == "lenet":
        assert dataset == "cifar10"
        from models import LeNet

        model = LeNet(in_dim, num_classes).to(device)
    elif model_name == "minnet":
        assert dataset == "mnist"
        from models import MinNet

        model = MinNet().to(device)
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

    # Initiate optimizer
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
        random_first_epoch=not no_random_first_epoch,
        # Normalize the gradient
        normalize=normalize,
        # Random projection
        random_projection=random_projection,
        pi_eps=pi_eps,
        seed=seed,  # Only used for generating random projection
        # Probabilistic balance
        prob_balance=prob_balance,
        prob_balance_c=prob_c,
        ema_decay=ema_decay,
        # Other specific
        dtype=dtype,
        device=device,
        timer=timer,
        record_orders=True,
        record_herding=True,
        stale_mean_herding=False,
        cuda_herding=cuda_herding,
        record_norm=True,
    )

    # Initiate data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        persistent_workers=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        persistent_workers=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        persistent_workers=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # ft_compute_sample_grad = vmap(
    #     grad(partial(compute_loss, model, loss_fn), has_aux=True),
    #     in_dims=(None, None, 0, 0)
    # )  # the only argument of compute_loss is batched along the first axis

    ft_compute_sample_grad_and_loss = vmap(
        grad_and_value(partial(compute_loss, model, loss_fn), has_aux=True),
        in_dims=(None, None, 0, 0),
        randomness="different",
    )  # the only argument of compute_loss is batched along the first axis

    # Record the norms
    df_grad_norms = pd.DataFrame()

    # loop over the dataloader multiple times
    for epoch in range(epochs + 1):
        logs = {}

        # Only evaluate before the first epoch
        if epoch != 0:
            with timer(f"train"):
                # perform training (single loop over the train dataloader)
                train_loss = train(
                    train_loader=train_loader,
                    sampler=sampler,
                    params=params,
                    buffers=buffers,
                    ft_compute_sample_grad_and_loss=ft_compute_sample_grad_and_loss,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    metric=train_metric,
                    no_tqdm=no_tqdm,
                )
            train_acc = train_metric.compute()["accuracy"]
            grad_norms = sampler.sorter.grad_norms

            # Save the norms
            norm_mean = np.mean(grad_norms)
            norm_std = np.std(grad_norms)
            norm_max = np.max(grad_norms)
            df_grad_norms[epoch] = grad_norms

            herding = sampler.sorter.herding
            avg_grad_error = sampler.sorter.avg_grad_error

            # Update only after the first epoch
            logs.update(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_time": timer["train"][-1],
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                    "norm_max": norm_max,
                    "herding": herding,
                    "avg_grad_error": avg_grad_error,
                }
            )
        with timer("val"):
            train_eval_loss = validate(
                test_loader=train_eval_loader,
                model=model,
                loss_fn=loss_fn,
                metric=train_eval_metric,
                no_tqdm=no_tqdm,
            )
            # perform validation (single loop over the validation dataloader)
            val_loss = validate(
                test_loader=test_loader,
                model=model,
                loss_fn=loss_fn,
                metric=val_metric,
                no_tqdm=no_tqdm,
            )

        train_eval_acc = train_eval_metric.compute()["accuracy"]
        val_acc = val_metric.compute()["accuracy"]

        logs.update(
            {
                "train_eval_loss": train_eval_loss,
                "train_eval_accuracy": train_eval_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_time": timer["val"][-1],
            }
        )

        if epoch == 0:
            print(
                f"Before training | "
                f"train_eval loss: {train_eval_loss :.3f} "
                f"acc : {train_eval_acc:.3f} | "
                f"val_loss: {val_loss :.3f} "
                f"val_acc: {train_eval_acc:.3f} | "
                f'val: {pretty_time(timer["val"][-1])}'
            )
        else:
            print(
                f"Epoch: {epoch} | "
                f"train loss: {train_loss :.3f} "
                f"acc: {train_acc:.3f} | "
                f"train_eval loss: {train_eval_loss :.3f} "
                f"acc : {train_eval_acc:.3f} | "
                f"val_loss: {val_loss :.3f} "
                f"val_acc: {train_eval_acc:.3f} | "
                f"norm mean: {norm_mean:.2f} "
                f"std: {norm_std:.2f} "
                f"max: {norm_max:.2f} | "
                f'train: {pretty_time(timer["train"][-1])} '
                f'val: {pretty_time(timer["val"][-1])} | '
                f"herding: {herding:.2f} "
                f"avg_grad_error: {avg_grad_error:.2f}"
            )

        # save checkpoint
        if save_checkpoints:
            checkpoint_name = exp_id + f"_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path / checkpoint_name)

        if epoch > 0:
            # Save the grad norms
            df_grad_norms.describe().to_csv(
                checkpoint_path / f"{exp_id}_{epochs}_grad_norms_proc.csv"
            )
            # Save the timer
            timer.save(checkpoint_path / f"{exp_id}_{epochs}_timer_proc.pt")
            timer.summary().to_csv(
                checkpoint_path / f"{exp_id}_{epochs}_timer_proc.csv"
            )

        # Save the orders
        torch.save(
            torch.tensor(sampler.orders_history),
            checkpoint_path / f"{exp_id}_{epochs}_orders_proc.pt",
        )
        wandb.log(logs)

    print(torch.cuda.memory_summary())

    print("-" * 50)
    print(df_grad_norms.describe())

    print("-" * 50)
    print("Timer:")
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

    # Save the grad norms
    df_grad_norms.describe().to_csv(
        checkpoint_path / f"{exp_id}_{epochs}_grad_norms.csv"
    )
    # Save the timer
    timer.save(checkpoint_path / f"{exp_id}_{epochs}_timer.pt")
    timer.summary().to_csv(checkpoint_path / f"{exp_id}_{epochs}_timer.csv")

    # Save the orders
    torch.save(
        torch.tensor(sampler.orders_history),
        checkpoint_path / f"{exp_id}_{epochs}_orders.pt",
    )


if __name__ == "__main__":
    cli()
