#!/usr/bin/env python
# coding: utf-8
# Gary Wei github.com/garywei944

import evaluate
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm, trange
from absl import logging
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import set_seed, HfArgumentParser

from grabsampler import GraBSampler, BalanceType
from grabsampler.utils import EventTimer, pretty_time

from cd2root import cd2root

cd2root()

from experiments.cv.cv import Args, get_exp_id, get_dataset, get_model

from experiments.utils.arguments import GraBArgs, TrainArgs

DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
MODELS = ["lr", "lenet", "resnet", "minnet"]


def get_optimizer(train_args: TrainArgs, model: nn.Module) -> torch.optim.Optimizer:
    # Initiate optimizer
    if train_args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_args.learning_rate,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay,
        )
    elif train_args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            weight_decay=train_args.weight_decay,
        )
    else:
        raise ValueError("Unknown optimizer")

    return optimizer


def train(
    train_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: GraBSampler,
    metric: evaluate.EvaluationModule,
    pbar: tqdm,
    device: torch.device = torch.device("cuda"),
):
    model.train()
    running_loss, n = 0, 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        b = x.shape[0]
        n += b

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        grads = {k: v.grad for k, v in model.named_parameters() if v.grad is not None}

        sampler.step(grads, batch=False, batch_gradient_size=b)

        running_loss += loss.item() * len(x)
        metric.add_batch(predictions=logits.argmax(dim=-1), references=y)

        pbar.update(1)

    return running_loss / n


# validation function
@torch.no_grad()
def validate(
    test_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    metric: evaluate.EvaluationModule,
    no_tqdm: bool = False,
    device: torch.device = torch.device("cuda"),
):
    model.eval()
    running_loss, n = 0, 0
    # look over the validation dataloader
    for x, y in tqdm(test_loader, leave=False, disable=no_tqdm):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        running_loss += loss.item() * x.shape[0]
        n += x.shape[0]
        metric.add_batch(predictions=outputs.argmax(dim=-1), references=y)
    return running_loss / n


def main():
    args: Args
    grab_args: GraBArgs
    train_args: TrainArgs
    args, grab_args, train_args = HfArgumentParser(
        (Args, GraBArgs, TrainArgs)
    ).parse_args_into_dataclasses()

    # Init wandb
    config = {
        **vars(args),
        **vars(grab_args),
        **vars(train_args),
    }
    config['batch_grad'] = True
    wandb.init(
        project=f"grab-{args.dataset_name}"
        if train_args.wandb_project is None
        else train_args.wandb_project,
        entity="grab",
        mode="online" if train_args.wandb else "offline",
        config=config,
    )

    # Set up exp_id and checkpoint path
    exp_id = get_exp_id(args, grab_args, train_args)
    checkpoint_path = (
        train_args.output_path
        / args.dataset_name
        / args.model_name
        / str(grab_args.balance_type)
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Set up device, dtype, seed, logging, and timer
    device = train_args.device
    if train_args.seed is not None:
        set_seed(train_args.seed)
    logging.set_verbosity(logging.INFO)
    timer = EventTimer(device=device)

    # Set dtype, only used for the sampler
    logging.info(f"Experiment ID: {exp_id}")
    # print(tabulate(sorted(list(config.items()))))
    print(tabulate(config.items()))

    # Load dataset
    train_dataset, test_dataset, in_dim, num_classes = get_dataset(args)

    # Create metrics
    train_metric = evaluate.load("accuracy")
    train_eval_metric = evaluate.load("accuracy")
    val_metric = evaluate.load("accuracy")

    # Load the model
    # enforce that the same seed use the same model initialization
    if train_args.seed is not None:
        set_seed(train_args.seed)
    if args.model_name == "lr":
        assert args.dataset_name in ["mnist", "fashion_mnist"]

        model = nn.Linear(in_dim, num_classes).to(device)
    elif args.model_name == "lenet":
        assert args.dataset_name in ["cifar10", "cifar100"]
        from models import LeNet

        model = LeNet(in_dim, num_classes).to(device)
    elif args.model_name == "resnet":
        assert args.dataset_name == "cifar10"
        from models import ResNet

        model = ResNet(
            depth=args.resnet_depth, num_classes=num_classes, norm_type="in"
        ).to(device)
    elif args.model_name == "minnet":
        assert args.dataset_name in ["mnist", "fashion_mnist"]
        from models import MinNet

        model = MinNet().to(device)
    else:
        raise ValueError

    loss_fn = nn.CrossEntropyLoss()

    optimizer = get_optimizer(train_args, model)

    # Initiate sampler
    d = sum(p[1].numel() for p in model.named_parameters())
    logging.info(f"Number of training examples: n = {len(train_dataset):,}")
    logging.info(f"Number of parameters: d = {d:,}")

    # Set up the sampler
    sampler = GraBSampler(
        train_dataset,
        trainable_params=dict(model.named_parameters()),
        # Probabilistic balance
        # Other specific
        timer=timer,
        record_herding=grab_args.report_grads,
        stale_mean_herding=False,
        cuda_herding=not grab_args.cpu_herding,
        record_norm=grab_args.report_grads,
        **config,
    )

    # Initiate data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args.train_batch_size,
        sampler=sampler,
        persistent_workers=False,
        num_workers=train_args.num_workers,
    )
    train_eval_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args.eval_batch_size,
        persistent_workers=False,
        num_workers=train_args.num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_args.eval_batch_size,
        persistent_workers=False,
        num_workers=train_args.num_workers,
    )
    # Train
    pbar = trange(
        len(train_loader) * train_args.epochs,
        leave=False,
        disable=not train_args.tqdm,
    )
    for epoch in range(0 if train_args.log_first_step else 1, train_args.epochs + 1):
        logs = {
            "epoch": epoch,
            "iteration": epoch * len(train_loader),
        }

        if epoch != 0:
            with timer("train"):
                train_loss = train(
                    train_loader=train_loader,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    sampler=sampler,
                    metric=train_metric,
                    pbar=pbar,
                    device=device,
                )
            train_acc = train_metric.compute()["accuracy"]
            logs.update(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_time": timer["train"][-1],
                }
            )
        with timer("val"):
            train_eval_loss = validate(
                test_loader=train_eval_loader,
                model=model,
                loss_fn=loss_fn,
                metric=train_eval_metric,
                no_tqdm=not train_args.tqdm,
                device=device,
            )
            # perform validation (single loop over the validation dataloader)
            val_loss = validate(
                test_loader=test_loader,
                model=model,
                loss_fn=loss_fn,
                metric=val_metric,
                no_tqdm=not train_args.tqdm,
                device=device,
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
            tqdm.write(
                f"Before training | "
                f"train_eval loss: {train_eval_loss :.3f} "
                f"acc : {train_eval_acc:.3f} | "
                f"val_loss: {val_loss :.3f} "
                f"val_acc: {val_acc:.3f} | "
                f'val: {pretty_time(timer["val"][-1])}'
            )
        else:
            log_msg = (
                f"Epoch: {epoch} | "
                f"train loss: {train_loss :.3f} "
                f"acc: {train_acc:.3f} | "
                f"train_eval loss: {train_eval_loss :.3f} "
                f"acc : {train_eval_acc:.3f} | "
                f"val loss: {val_loss :.3f} "
                f"acc: {val_acc:.3f} | "
                f'train: {pretty_time(timer["train"][-1])} '
                f'val: {pretty_time(timer["val"][-1])}'
            )
            tqdm.write(log_msg)

        # save checkpoint
        if train_args.save_strategy == "epoch":
            checkpoint_name = exp_id + f"_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path / checkpoint_name)

        if epoch > 0:
            # Save the timer
            timer.save(checkpoint_path / f"{exp_id}_{train_args.epochs}_timer_proc.pt")
            timer.summary().to_csv(
                checkpoint_path / f"{exp_id}_{train_args.epochs}_timer_proc.csv"
            )

        # Log to wandb
        wandb.log(logs, step=epoch)

    pbar.close()
    print(torch.cuda.memory_summary())

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

    # Save the timer
    timer.save(checkpoint_path / f"{exp_id}_{train_args.epochs}_timer.pt")
    timer.summary().to_csv(checkpoint_path / f"{exp_id}_{train_args.epochs}_timer.csv")


if __name__ == "__main__":
    main()
