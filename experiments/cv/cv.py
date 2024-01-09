#!/usr/bin/env python
# coding: utf-8
# Gary Wei github.com/garywei944

from functools import partial
from typing import Callable

import evaluate
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm, trange
from absl import logging
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from transformers import set_seed, HfArgumentParser
from dataclasses import dataclass, field

import torchopt
from torch.func import grad_and_value, vmap, functional_call
from torchopt.typing import GradientTransformation

from grabsampler import GraBSampler, BalanceType
from grabsampler.utils import EventTimer, pretty_time

from torchopt.schedule import linear_schedule

from cd2root import cd2root

cd2root()

from experiments.utils.func_helpers import make_func_params
from experiments.utils.arguments import GraBArgs, TrainArgs

DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
MODELS = ["lr", "lenet", "resnet", "minnet", "wrn"]


@dataclass
class Args:
    dataset_name: str = field(
        default="mnist",
        metadata={
            "aliases": ["-d"],
            "help": "Which dataset to use",
            "choices": DATASETS,
        },
    )
    model_name: str = field(
        default="lr",
        metadata={
            "aliases": ["-model"],
            "help": "Which model to use",
            "choices": MODELS,
        },
    )
    num_train_examples: int = field(
        default=None,
        metadata={
            "aliases": ["-ntr"],
            "help": "Number of samples to use for training",
        },
    )
    num_test_examples: int = field(
        default=None,
        metadata={
            "aliases": ["-nte"],
            "help": "Number of samples to use for testing",
        },
    )
    resnet_depth: int = field(
        default=8,
        metadata={
            "help": "Depth of ResNet",
        },
    )
    wrn_norm: str = field(
        default="gn",
        metadata={
            "choices": ["in", "bn", "gn"],
            "help": "Norm type for ResNet",
        },
    )
    data_augmentation: str = field(
        default="none",
        metadata={
            "aliases": ["-da"],
            "choices": ["none", "basic"],
            "help": "Data augmentation",
        },
    )


def multi_step_lr(
    learning_rate: float,
    milestones: list[int],
    gamma: float = 0.1,
):
    def _multi_step_lr(step: int):
        return learning_rate * gamma ** sum(step > m for m in milestones)

    return _multi_step_lr


def get_exp_id(args: Args, grab_args: GraBArgs, train_args: TrainArgs) -> str:
    # Unique experiment name for checkpoints
    exp_id = (
        f"{args.dataset_name}_{args.model_name}_{grab_args.balance_type}"
        f"_{train_args.optimizer}_lr_{train_args.learning_rate}"
        f"_wd_{train_args.weight_decay}"
        f"_b_{train_args.train_batch_size}_seed_{train_args.seed}"
    )

    if grab_args.normalize_grad:
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


def get_dataset(args: Args) -> tuple[Dataset, Dataset, int, int]:
    # Load the dataset
    if args.dataset_name == "mnist":
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
    elif args.dataset_name == "fashion_mnist":
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
    elif args.dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )

        if args.data_augmentation == "basic":
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transform,
                ]
            )
        else:
            train_transform = transform

        # Loading the dataset and preprocessing
        train_dataset = datasets.CIFAR10(
            root="data/external", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root="data/external", train=False, download=True, transform=transform
        )

        in_dim, num_classes = 3, 10

    elif args.dataset_name == "cifar100":
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
    else:
        raise ValueError

    # Use a subset of training examples
    if args.num_train_examples is not None:
        train_dataset = Subset(train_dataset, range(args.num_train_examples))
    if args.num_test_examples is not None:
        test_dataset = Subset(test_dataset, range(args.num_test_examples))

    return train_dataset, test_dataset, in_dim, num_classes


def get_model(
    args: Args, train_args: TrainArgs, in_dim: int, num_classes: int
) -> tuple[nn.Module, dict[str, nn.Parameter], dict[str, Tensor], nn.Module]:
    device = train_args.device

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
        from models.preact_resnet import PreActResNet18GroupNorm

        # model = ResNet(
        #     depth=args.resnet_depth, num_classes=num_classes, norm_type="in"
        # ).to(device)
        model = PreActResNet18GroupNorm(n_cls=num_classes).to(device)
    elif args.model_name == "minnet":
        assert args.dataset_name in ["mnist", "fashion_mnist"]
        from models import MinNet

        model = MinNet().to(device)
    elif args.model_name == "wrn":
        assert args.dataset_name in ["cifar10", "cifar100"]
        from models import WRN

        model = WRN(
            input_shape=(1, 3, 32, 32),
            n_classes=num_classes,
            norm=args.wrn_norm,
            gn_groups=8,
        ).to(device)
    else:
        raise ValueError
    # Transform everything to functional programming

    # Get params
    params, buffers = make_func_params(model)
    loss_fn = nn.CrossEntropyLoss()

    return model, params, buffers, loss_fn


def get_optimizer(
    train_args: TrainArgs, milestones: list[int], gamma: float = 0.1
) -> tuple[GradientTransformation, (Callable | float)]:
    if train_args.scheduler == "constant":
        lr = lambda x: train_args.learning_rate
    elif train_args.scheduler == "multi_step_lr":
        lr = multi_step_lr(
            train_args.learning_rate,
            milestones,
            gamma=gamma,
        )
    else:
        raise ValueError("Unknown scheduler")

    # Initiate optimizer
    if train_args.optimizer == "sgd":
        optimizer = torchopt.sgd(
            lr,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay,
            nesterov=True,
        )
    elif train_args.optimizer in ["adam", "adamw"]:
        optimizer = torchopt.adamw(
            lr,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            weight_decay=train_args.weight_decay,
            use_accelerated_op=True,
        )
    else:
        raise ValueError("Unknown optimizer")

    return optimizer, lr


def compute_loss(
    model: nn.Module,
    loss_fn: nn.Module,
    params: dict[str, nn.Parameter],
    buffers: dict[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
):
    inputs = inputs.unsqueeze(0)
    targets = targets.unsqueeze(0)

    logits = functional_call(model, (params, buffers), (inputs,))

    return loss_fn(logits, targets), logits


@torch.no_grad()
def train(
    train_loader: DataLoader,
    sampler: GraBSampler,
    params: dict[str, nn.Parameter],
    buffers: dict[str, Tensor],
    ft_compute_sample_grad_and_loss: callable,
    optimizer: GradientTransformation,
    opt_state: dict[str, Tensor],
    metric: evaluate.EvaluationModule,
    pbar: tqdm,
    device: torch.device = torch.device("cuda"),
    scheduler: callable = None,  # Only for debugging
) -> tuple[float, dict[str, nn.Parameter], dict[str, Tensor]]:
    running_loss, n = 0, 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        ft_per_sample_grads, (batch_loss, _) = ft_compute_sample_grad_and_loss(
            params, buffers, x, y
        )

        sampler.step(ft_per_sample_grads)

        grads = {k: g.mean(dim=0) for k, g in ft_per_sample_grads.items()}

        updates, opt_state = optimizer.update(
            grads, opt_state, params=params
        )  # get updates
        params = torchopt.apply_updates(params, updates)  # update network parameters

        if scheduler is not None:
            step = opt_state[-1].count[0].item()
            lr = scheduler(step)
            wandb.log({"lr": lr}, step=step)
            # print(f"step: {step} lr: {lr:.3e}")

        running_loss += batch_loss.sum().item()
        n += len(batch_loss)
        # metric.add_batch(predictions=logits.argmax(dim=-1), references=y)

        pbar.update(1)

    # TODO: maybe submit a github Issue that the count is not updated in opt_state
    return running_loss / n, params, opt_state


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
    config = {**vars(args), **vars(grab_args), **vars(train_args), "batch_grad": False}
    wandb.init(
        project=f"grab-{args.dataset_name}"
        if train_args.wandb_project is None
        else train_args.wandb_project,
        name=f"{args.model_name}-da_{args.data_augmentation}-func",
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

    # Initiate model
    model, params, buffers, loss_fn = get_model(args, train_args, in_dim, num_classes)

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
        batch_size=train_args.train_batch_size,
        # Probabilistic balance
        orders=orders,
        # Other specific
        timer=timer,
        record_herding=grab_args.report_grads,
        stale_mean_herding=False,
        cuda_herding=not grab_args.cpu_herding,
        record_norm=grab_args.report_grads,
        # For NTK
        model=model,
        params=params,
        buffers=buffers,
        loss_fn=loss_fn,
        dataset=train_dataset,
        kernel_dtype=torch.float32,
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

    # Initiate optimizer
    steps_per_epoch = len(train_loader)

    if args.model_name == "resnet":
        milestones = [
            0.5 * train_args.epochs * steps_per_epoch,
            0.75 * train_args.epochs * steps_per_epoch,
        ]
        gamma = 0.1
    elif args.model_name == "wrn":
        milestones = [
            60 * steps_per_epoch,
            120 * steps_per_epoch,
            160 * steps_per_epoch,
        ]
        gamma = 0.2
    else:
        milestones = [0]
        gamma = 1.0

    optimizer, scheduler = get_optimizer(
        train_args,
        milestones=milestones,
        gamma=gamma,
    )
    opt_state = optimizer.init(params)

    ft_compute_sample_grad_and_loss = vmap(
        grad_and_value(partial(compute_loss, model, loss_fn), has_aux=True),
        in_dims=(None, None, 0, 0),
        # randomness='different',
    )  # the only argument of compute_loss is batched along the first axis

    # Record the norms
    df_grad_norms = pd.DataFrame()

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
                train_loss, params, opt_state = train(
                    train_loader=train_loader,
                    sampler=sampler,
                    params=params,
                    buffers=buffers,
                    ft_compute_sample_grad_and_loss=ft_compute_sample_grad_and_loss,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    metric=train_metric,
                    pbar=pbar,
                    device=device,
                    scheduler=scheduler,
                )
            # train_acc = train_metric.compute()["accuracy"]
            # train_acc = 0.0
            logs.update(
                {
                    "train_loss": train_loss,
                    # "train_accuracy": train_acc,
                    "train_time": timer["train"][-1],
                }
            )

            if grab_args.report_grads:
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
                # f"acc: {train_acc:.3f} | "
                f"train_eval loss: {train_eval_loss :.3f} "
                f"acc : {train_eval_acc:.3f} | "
                f"val loss: {val_loss :.3f} "
                f"acc: {val_acc:.3f} | "
                f'train: {pretty_time(timer["train"][-1])} '
                f'val: {pretty_time(timer["val"][-1])}'
            )
            if grab_args.report_grads:
                log_msg += (
                    f" | norm_mean: {norm_mean:.2f} "
                    f"norm_std: {norm_std:.2f} "
                    f"norm_max: {norm_max:.2f} | "
                    f"herding: {herding:.2f} "
                    f"avg_grad_error: {avg_grad_error:.2f}"
                )
            tqdm.write(log_msg)

        # save checkpoint
        if train_args.save_strategy == "epoch":
            checkpoint_name = exp_id + f"_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path / checkpoint_name)

        if epoch > 0:
            if grab_args.report_grads:
                # Save the grad norms
                df_grad_norms.describe().to_csv(
                    checkpoint_path
                    / f"{exp_id}_{train_args.epochs}_grad_norms_proc.csv"
                )
            # Save the timer
            timer.save(checkpoint_path / f"{exp_id}_{train_args.epochs}_timer_proc.pt")
            timer.summary().to_csv(
                checkpoint_path / f"{exp_id}_{train_args.epochs}_timer_proc.csv"
            )

        # Save the orders
        if grab_args.record_orders:
            torch.save(
                torch.tensor(sampler.orders_history),
                checkpoint_path / f"{exp_id}_{train_args.epochs}_orders_proc.pt",
            )

        # Log to wandb
        wandb.log(logs, step=epoch)

    pbar.close()
    print(torch.cuda.memory_summary())

    if grab_args.report_grads:
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
    timer.save(checkpoint_path / f"{exp_id}_{train_args.epochs}_timer.pt")
    timer.summary().to_csv(checkpoint_path / f"{exp_id}_{train_args.epochs}_timer.csv")
    if grab_args.report_grads:
        # Save the grad norms
        df_grad_norms.describe().to_csv(
            checkpoint_path / f"{exp_id}_{train_args.epochs}_grad_norms.csv"
        )

    if grab_args.record_orders:
        # Save the orders
        torch.save(
            torch.tensor(sampler.orders_history),
            checkpoint_path / f"{exp_id}_{train_args.epochs}_orders.pt",
        )


if __name__ == "__main__":
    main()
