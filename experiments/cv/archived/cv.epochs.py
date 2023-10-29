#!/usr/bin/env python
# coding: utf-8

from functools import partial

import evaluate
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from absl import logging
from tabulate import tabulate
from typing import Literal

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from transformers import set_seed

import torchopt
from torch.func import grad, grad_and_value, vmap, functional_call

from grabsampler import GraBSampler, BalanceType
from grabsampler.utils import EventTimer, pretty_time

from cd2root import cd2root

cd2root()

from experiments.utils.func_helpers import make_func_params
from experiments.utils.arguments import GraBArguments, TrainArgs

DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
MODELS = ["lr", "lenet", "resnet", "minnet"]


class Args(GraBArguments, TrainArgs):
    dataset_name: Literal[tuple(DATASETS)] = "mnist"  # Which dataset to use
    model_name: Literal[tuple(MODELS)] = "lr"  # Which model to use
    num_train_examples: int = None  # Number of samples to use for training
    num_test_examples: int = None  # Number of samples to use for testing
    resnet_depth: int = 8  # Depth of ResNet

    def configure(self) -> None:
        GraBArguments.configure(self)
        TrainArgs.configure(self)

        self.add_argument("-d", "--dataset_name")
        self.add_argument("-model", "--model_name")
        self.add_argument("-ntr", "--num_train_examples")
        self.add_argument("-nte", "--num_test_examples")

    def process_args(self) -> None:
        GraBArguments.process_args(self)
        TrainArgs.process_args(self)


def compute_loss(model, loss_fn, params, buffers, inputs, targets):
    print(type(model))
    print(type(loss_fn))
    print(type(params))
    print(type(params['bias']))

    print(type(buffers))
    print(type(inputs))
    print(type(targets))

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
    running_loss, n = 0, 0
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

        running_loss += batch_loss.sum().item()
        n += len(batch_loss)
        metric.add_batch(predictions=logits.argmax(dim=-1), references=y)

    return running_loss / n


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
    running_loss, n = 0, 0
    # look over the validation dataloader
    for _, (x, y) in tqdm(
        enumerate(test_loader), total=len(test_loader), leave=False, disable=no_tqdm
    ):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        running_loss += loss.item() * x.shape[0]
        n += x.shape[0]
        metric.add_batch(predictions=outputs.argmax(dim=-1), references=y)
    return running_loss / n


def main():
    args = Args().parse_args()

    # Initialize a wandb run
    wandb.init(
        project=f"grab-{args.dataset_name}"
        if args.wandb_project is None
        else args.wandb_project,
        entity="grab",
        mode="online" if args.wandb else "offline",
        config=args.as_dict(),
    )

    # Set up exp_id and checkpoint path
    exp_id = get_exp_id(args)
    checkpoint_path = (
        args.output_path / args.dataset_name / args.model_name / str(args.balance_type)
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Set up device, dtype, seed, logging, and timer
    device = args.device
    dtype = args.dtype
    if args.seed is not None:
        set_seed(args.seed)
    logging.set_verbosity(logging.INFO)
    timer = EventTimer(device=device)

    # Set dtype, only used for the sampler
    logging.info(f"Experiment ID: {exp_id}")
    print(tabulate(args.as_dict().items()))

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

        loss_fn = nn.CrossEntropyLoss().to(device)
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

        loss_fn = nn.CrossEntropyLoss().to(device)
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

        # Loading the dataset and preprocessing
        train_dataset = datasets.CIFAR10(
            root="data/external", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="data/external", train=False, download=True, transform=transform
        )

        in_dim, num_classes = 3, 10

        loss_fn = nn.CrossEntropyLoss().to(device)
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

        loss_fn = nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError

    # Use a subset of training examples
    if args.num_train_examples is not None:
        train_dataset = Subset(train_dataset, range(args.num_train_examples))
    if args.num_test_examples is not None:
        test_dataset = Subset(test_dataset, range(args.num_test_examples))

    # Create metrics
    train_metric = evaluate.load("accuracy")
    train_eval_metric = evaluate.load("accuracy")
    val_metric = evaluate.load("accuracy")

    # Load the model
    # enforce that the same seed use the same model initialization
    if args.seed is not None:
        set_seed(args.seed)
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
    # Transform everything to functional programming

    # Get params
    params, buffers = make_func_params(model)

    # Initiate optimizer
    if args.optimizer == "sgd":
        optimizer = torchopt.sgd(
            args.learning_rate,
            momentum=args.adam_beta1,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torchopt.adamw(
            args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
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
    if args.order_path is not None:
        orders = torch.load(args.order_path)
        if len(orders.shape) == 2:
            orders = orders[-1].tolist()
        else:
            orders = orders.tolist()

    sampler = GraBSampler(
        train_dataset,
        params,
        # Probabilistic balance
        orders=orders,
        # Other specific
        timer=timer,
        record_herding=args.report_grads,
        stale_mean_herding=False,
        cuda_herding=not args.cpu_herding,
        record_norm=args.report_grads,
        # For NTK
        model=model,
        params=params,
        buffers=buffers,
        loss_fn=loss_fn,
        dataset=train_dataset,
        kernel_dtype=torch.float32,
        **args.as_dict(),
    )

    # Initiate data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        persistent_workers=False,
        num_workers=args.num_workers,
    )
    train_eval_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.eval_batch_size,
        persistent_workers=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        persistent_workers=False,
        num_workers=args.num_workers,
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
    epochs = int(args.epochs)

    for epoch in range(0 if args.log_first_step else 1, epochs + 1):
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
                    no_tqdm=not args.tqdm,
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

            if args.report_grads:
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
                no_tqdm=not args.tqdm,
                device=device,
            )
            # perform validation (single loop over the validation dataloader)
            val_loss = validate(
                test_loader=test_loader,
                model=model,
                loss_fn=loss_fn,
                metric=val_metric,
                no_tqdm=not args.tqdm,
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
            if args.report_grads:
                log_msg += (
                    f" | norm_mean: {norm_mean:.2f} "
                    f"norm_std: {norm_std:.2f} "
                    f"norm_max: {norm_max:.2f} | "
                    f"herding: {herding:.2f} "
                    f"avg_grad_error: {avg_grad_error:.2f}"
                )
            print(log_msg)

        # save checkpoint
        if args.save_strategy == "epoch":
            checkpoint_name = exp_id + f"_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path / checkpoint_name)

        if epoch > 0:
            if args.report_grads:
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
        if args.record_orders:
            torch.save(
                torch.tensor(sampler.orders_history),
                checkpoint_path / f"{exp_id}_{epochs}_orders_proc.pt",
            )
        wandb.log(logs)

    print(torch.cuda.memory_summary())

    if args.report_grads:
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
    if args.report_grads:
        # Save the grad norms
        df_grad_norms.describe().to_csv(
            checkpoint_path / f"{exp_id}_{epochs}_grad_norms.csv"
        )

    if args.record_orders:
        # Save the orders
        torch.save(
            torch.tensor(sampler.orders_history),
            checkpoint_path / f"{exp_id}_{epochs}_orders.pt",
        )


def get_exp_id(args: Args):
    # Unique experiment name for checkpoints
    exp_id = (
        f"{args.dataset_name}_{args.model_name}_{args.balance_type}"
        f"_{args.optimizer}_lr_{args.learning_rate}"
        f"_wd_{args.weight_decay}"
        f"_b_{args.train_batch_size}_seed_{args.seed}"
    )

    if args.normalize_grad:
        exp_id += "_norm"
    if args.random_projection:
        exp_id += f"_pi_{args.random_projection_eps}"
    if args.prob_balance:
        exp_id += f"_prob_{args.prob_balance_c:.1f}"
    if args.balance_type in [
        BalanceType.RECURSIVE_BALANCE,
        BalanceType.RECURSIVE_PAIR_BALANCE,
    ]:
        exp_id += f"_depth_{args.depth}"
    if not args.random_first_epoch:
        exp_id += "_no_rr"
    if args.balance_type == BalanceType.EMA_BALANCE:
        exp_id += f"_ema_{args.ema_decay}"

    return exp_id


if __name__ == "__main__":
    main()
