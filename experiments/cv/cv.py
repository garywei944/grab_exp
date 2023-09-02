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

# Change pwd to the project root directory
PROJECT_NAME = "GraB-lib"
PROJECT_PATH = Path(__file__).resolve()
while PROJECT_PATH.name != PROJECT_NAME:
    PROJECT_PATH = PROJECT_PATH.parent
os.chdir(PROJECT_PATH)
sys.path.insert(0, str(PROJECT_PATH))

from experiments.utils.func_helpers import make_func_params
from experiments.utils.arguments import GraBArguments

DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
MODELS = ["lr", "lenet", "resnet", "minnet"]


@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # Task specific arguments
    dataset: str = field(
        default=DATASETS[0],
        metadata={"help": "Which dataset to use", "choices": DATASETS},
    )
    num_train_examples: int = field(
        default=None,
        metadata={"help": "Number of samples to use for training"},
    )
    num_test_examples: int = field(
        default=None,
        metadata={"help": "Number of samples to use for testing"},
    )
    model_name: str = field(
        default=MODELS[0],
        metadata={"help": "Which model to use", "choices": MODELS},
    )
    resnet_depth: int = field(
        default=8,
        metadata={
            "help": "Depth of ResNet",
        },
    )
    optimizer: str = field(
        default="sgd",
        metadata={
            "help": "Which optimizer to use",
            "choices": ["sgd", "adam", "adamw"],
        },
    )
    momentum: float = field(default=0.9, metadata={"help": "Momentum for SGD"})


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
    device: torch.device = torch.device("cuda"),
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

        running_loss += batch_loss.mean().item()
        if torch.isnan(batch_loss.mean()):
            raise ValueError
        metric.add_batch(predictions=logits.argmax(dim=-1), references=y)

    return running_loss / len(train_loader)


# validation function
@torch.no_grad()
def validate(
    test_loader,
    model,
    loss_fn,
    metric: evaluate.EvaluationModule,
    no_tqdm=False,
    device: torch.device = torch.device("cuda"),
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((Arguments, GraBArguments, TrainingArguments))

    # Only for PyCharm type hint
    args: Arguments
    grab_args: GraBArguments
    training_args: TrainingArguments
    args, grab_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize a wandb run
    wandb.init(
        project=f"grab-{args.dataset}"
        if grab_args.wandb_project is None
        else grab_args.wandb_project,
        entity="grab",
        mode="online" if grab_args.use_wandb else "offline",
        config={
            **vars(args),
            **vars(grab_args),
            **vars(training_args),
        },
    )

    # Set up exp_id and checkpoint path
    exp_id = get_exp_id(args, grab_args, training_args)
    checkpoint_path = (
        Path(training_args.output_dir)
        / args.dataset
        / args.model_name
        / grab_args.balance_type
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Set up device, dtype, seed, logging, and timer
    device = training_args.device
    if training_args.fp16:
        dtype = torch.float16
    elif training_args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    set_seed(training_args.seed)
    logging.set_verbosity(logging.INFO)
    timer = EventTimer(device=device)

    # Set dtype, only used for the sampler
    logging.info(f"Experiment ID: {exp_id}")
    pprint(dict(wandb.config))

    # Load the dataset
    if args.dataset == "mnist":
        if args.model_name == "lr":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.view(-1)),
                ]
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        train_dataset = datasets.MNIST(
            root="data/external", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="data/external", train=False, transform=transform
        )

        in_dim, num_classes = 784, 10

        loss_fn = nn.CrossEntropyLoss().to(device)
    elif args.dataset == "fashion_mnist":
        # https://www.kaggle.com/code/leifuer/intro-to-pytorch-fashion-mnist
        # Define a transform to normalize the data
        if args.model_name == "lr":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                    transforms.Lambda(lambda x: x.view(-1)),
                ]
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
        # Download and load the training data
        train_dataset = datasets.FashionMNIST(
            "data/external", download=True, train=True, transform=transform
        )
        # Download and load the test data
        test_dataset = datasets.FashionMNIST(
            "data/external", download=True, train=False, transform=transform
        )

        in_dim, num_classes = 784, 10

        loss_fn = nn.CrossEntropyLoss().to(device)
    elif args.dataset == "cifar10":
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

        in_dim, num_classes = 3, 10

        loss_fn = nn.CrossEntropyLoss().to(device)
    elif args.dataset == "cifar100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.50707516, 0.48654887, 0.44091784],
                    std=[0.26733429, 0.25643846, 0.27615047],
                ),
            ]
        )

        # Loading the dataset and preprocessing
        train_dataset = datasets.CIFAR100(
            root="data/external", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root="data/external", train=False, download=True, transform=transform
        )

        in_dim, num_classes = 3, 100

        loss_fn = nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError

    # Use a subset of training examples
    if args.num_train_examples is not None:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(args.num_train_examples)
        )
    if args.num_test_examples is not None:
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(args.num_test_examples)
        )

    # Create metrics
    train_metric = evaluate.load("accuracy")
    train_eval_metric = evaluate.load("accuracy")
    val_metric = evaluate.load("accuracy")

    # Load the model
    # enforce that the same seed use the same model initialization
    set_seed(training_args.seed)
    if args.model_name == "lr":
        assert args.dataset in ["mnist", "fashion_mnist"]

        model = nn.Linear(in_dim, num_classes).to(device)
    elif args.model_name == "lenet":
        assert args.dataset in ["cifar10", "cifar100"]
        from models import LeNet

        model = LeNet(in_dim, num_classes).to(device)
    elif args.model_name == "resnet":
        assert args.dataset == "cifar10"
        from models import ResNet

        model = ResNet(
            depth=args.resnet_depth, num_classes=num_classes, norm_type="in"
        ).to(device)
    elif args.model_name == "minnet":
        assert args.dataset in ["mnist", "fashion_mnist"]
        from models import MinNet

        model = MinNet().to(device)
    else:
        raise ValueError
    # Transform everything to functional programming

    # Get params
    params, buffers = make_func_params(model)

    # Initiate optimizer
    if args.optimizer == "sgd":
        optimizer = torchopt.sgd(
            training_args.learning_rate,
            momentum=args.momentum,
            weight_decay=training_args.weight_decay,
        )
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torchopt.adamw(
            training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            use_accelerated_op=True,
        )
    else:
        raise ValueError("Unknown optimizer")

    opt_state = optimizer.init(params)  # init optimizer

    # Initiate sampler
    d = sum(p[1].numel() for p in model.named_parameters())
    logging.info(f"Number of training examples: n = {len(train_dataset):,}")
    logging.info(f"Number of parameters: d = {d:,}")

    # Load orders for FixedOrdering
    orders = None
    if grab_args.order_path is not None:
        orders = torch.load(grab_args.order_path)
        if len(orders.shape) == 2:
            orders = orders[-1].tolist()
        else:
            orders = orders.tolist()

    sampler = GraBSampler(
        train_dataset,
        params,
        batch_size=training_args.train_batch_size,
        # Random projection
        seed=training_args.seed,  # Only used for generating random projection
        # Probabilistic balance
        orders=orders,
        # Other specific
        dtype=dtype,
        device=device,
        timer=timer,
        record_herding=grab_args.record_grads,
        stale_mean_herding=False,
        cuda_herding=not grab_args.cpu_herding,
        record_norm=grab_args.record_grads,
        # For NTK
        model=model,
        params=params,
        buffers=buffers,
        loss_fn=loss_fn,
        dataset=train_dataset,
        kernel_dtype=torch.float32,
        **vars(grab_args),
    )

    # Initiate data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=training_args.train_batch_size,
        sampler=sampler,
        persistent_workers=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=training_args.eval_batch_size,
        persistent_workers=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=training_args.eval_batch_size,
        persistent_workers=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    # ft_compute_sample_grad = vmap(
    #     grad(partial(compute_loss, model, loss_fn), has_aux=True),
    #     in_dims=(None, None, 0, 0)
    # )  # the only argument of compute_loss is batched along the first axis

    ft_compute_sample_grad_and_loss = vmap(
        grad_and_value(partial(compute_loss, model, loss_fn), has_aux=True),
        in_dims=(None, None, 0, 0),
        # randomness='different',
    )  # the only argument of compute_loss is batched along the first axis

    # Record the norms
    df_grad_norms = pd.DataFrame()

    # loop over the dataloader multiple times
    epochs = int(training_args.num_train_epochs)

    for epoch in range(0 if training_args.logging_first_step else 1, epochs + 1):
        logs = {}

        # Only evaluate before the first epoch
        if epoch != 0:
            # compute orders if we are using ntk balance
            if grab_args.balance_type == BalanceType.NTK_EIGEN.value:
                sampler.sorter.compute_order(
                    model=model,
                    params=params,
                    buffers=buffers,
                    loss_fn=loss_fn,
                )

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
                    no_tqdm=training_args.disable_tqdm,
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

            if grab_args.record_grads:
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
                no_tqdm=training_args.disable_tqdm,
                device=device,
            )
            # perform validation (single loop over the validation dataloader)
            val_loss = validate(
                test_loader=test_loader,
                model=model,
                loss_fn=loss_fn,
                metric=val_metric,
                no_tqdm=training_args.disable_tqdm,
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
            print(
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
            if grab_args.record_grads:
                log_msg += (
                    f" | norm_mean: {norm_mean:.2f} "
                    f"norm_std: {norm_std:.2f} "
                    f"norm_max: {norm_max:.2f} | "
                    f"herding: {herding:.2f} "
                    f"avg_grad_error: {avg_grad_error:.2f}"
                )
            print(log_msg)

        # save checkpoint
        if training_args.save_strategy == "epoch":
            checkpoint_name = exp_id + f"_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path / checkpoint_name)

        if epoch > 0:
            if grab_args.record_grads:
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
        if grab_args.record_orders:
            torch.save(
                torch.tensor(sampler.orders_history),
                checkpoint_path / f"{exp_id}_{epochs}_orders_proc.pt",
            )
        wandb.log(logs)

    print(torch.cuda.memory_summary())

    if grab_args.record_grads:
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

    # Save the timer
    timer.save(checkpoint_path / f"{exp_id}_{epochs}_timer.pt")
    timer.summary().to_csv(checkpoint_path / f"{exp_id}_{epochs}_timer.csv")
    if grab_args.record_grads:
        # Save the grad norms
        df_grad_norms.describe().to_csv(
            checkpoint_path / f"{exp_id}_{epochs}_grad_norms.csv"
        )

    if grab_args.record_orders:
        # Save the orders
        torch.save(
            torch.tensor(sampler.orders_history),
            checkpoint_path / f"{exp_id}_{epochs}_orders.pt",
        )


def get_exp_id(
    args: Arguments, grab_args: GraBArguments, training_args: TrainingArguments
):
    # Unique experiment name for checkpoints
    exp_id = (
        f"{args.dataset}_{args.model_name}_{grab_args.balance_type}"
        f"_{args.optimizer}_lr_{training_args.learning_rate}"
        f"_wd_{training_args.weight_decay}"
        f"_b_{training_args.train_batch_size}_seed_{training_args.seed}"
    )

    if grab_args.normalize_grads:
        exp_id += "_norm"
    if grab_args.random_projection:
        exp_id += f"_pi_{grab_args.random_projection_eps}"
    if grab_args.prob_balance:
        exp_id += f"_prob_{grab_args.prob_balance_c:.1f}"
    if grab_args.balance_type in [
        BalanceType.RECURSIVE_BALANCE,
        BalanceType.RECURSIVE_PAIR_BALANCE,
    ]:
        exp_id += f"_depth_{grab_args.depth}"
    if not grab_args.random_first_epoch:
        exp_id += "_no_rr"
    if grab_args.balance_type == BalanceType.EMA_BALANCE:
        exp_id += f"_ema_{grab_args.ema_decay}"

    return exp_id


if __name__ == "__main__":
    main()
