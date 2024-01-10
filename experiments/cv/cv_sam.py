#!/usr/bin/env python
# coding: utf-8
# Gary Wei github.com/garywei944

from functools import partial

import evaluate
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm, trange
from absl import logging
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
from torchvision import datasets, transforms
from transformers import set_seed, HfArgumentParser
from dataclasses import dataclass, field

import torchopt
from torch.func import grad_and_value, vmap, functional_call
from torchopt.typing import GradientTransformation

from grabsampler import GraBSampler, BalanceType
from grabsampler.utils import EventTimer, pretty_time, StaleMeanEstimator

from cd2root import cd2root

cd2root()

from experiments.utils.func_helpers import make_func_params
from experiments.utils.arguments import GraBArgs, TrainArgs
from experiments.cv.cv import (
    get_optimizer,
    get_dataset,
    compute_loss,
    get_model,
    get_exp_id,
    validate,
)

DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
MODELS = ["lr", "lenet", "resnet", "minnet", "wrn"]


class DESampler(Sampler):
    orders: Tensor
    next_orders: Tensor
    acc: Tensor

    def __init__(self, n: int, d: int, *args, **kwargs):
        super().__init__()
        self.n = n
        self.d = d

        self.orders: Tensor = torch.randperm(n, dtype=torch.int64)
        # self.orders = torch.arange(n, dtype=torch.int64)
        self.next_orders: Tensor = self.orders.clone()

        self.acc = torch.zeros(d, dtype=torch.float32, device=torch.device("cuda"))

        self.idx = self.n
        self.left = self.n
        self.right = self.n - 1

        # logging.warning("Doing reversed GraB! Just for experiment, change this!")

    # # pair balance not working well
    # def step(
    #     self,
    #     grads1: dict[str, Tensor],
    #     grads2: dict[str, Tensor],
    #     indices: Tensor,
    # ):
    #     b = indices.shape[1]
    #     grad1 = torch.cat([grads1[k].flatten() for k in grads1])
    #     grad2 = torch.cat([grads2[k].flatten() for k in grads2])
    #
    #     diff = grad1 - grad2
    #     if torch.inner(diff, self.acc) < 0:
    #         self.next_orders[self.left : self.left + b].copy_(
    #             self.orders[indices[0] + self.idx]
    #         )
    #         self.next_orders[self.right - b + 1 : self.right + 1].copy_(
    #             self.orders[indices[1] + self.idx]
    #         )
    #         self.acc.add_(diff)
    #     else:
    #         self.next_orders[self.right - b + 1 : self.right + 1].copy_(
    #             self.orders[indices[0] + self.idx]
    #         )
    #         self.next_orders[self.left : self.left + b].copy_(
    #             self.orders[indices[1] + self.idx]
    #         )
    #         self.acc.sub_(diff)
    #     self.left += b
    #     self.right -= b
    #     self.idx += b * 2

    def compute_sings(self, grads: dict[str, Tensor]):
        b = next(iter(grads.values())).shape[0]

        assert b % 2 == 0
        acc = self.acc.clone()

        grads = torch.cat([v.reshape(b, -1) for k, v in grads.items()], dim=1)

        pair_grad = grads[::2] - grads[1::2]

        signs = []

        for i in range(b // 2):
            if torch.inner(pair_grad[i], acc) < 0:
                signs.append(True)
                acc.add_(pair_grad[i])
            else:
                signs.append(False)
                acc.sub_(pair_grad[i])

        return signs

    def step(
        self,
        grads: dict[str, Tensor],
        signs: list[bool],
    ):
        b = next(iter(grads.values())).shape[0]

        assert b % 2 == 0
        assert len(signs) == b // 2

        grads = torch.cat([v.reshape(b, -1) for k, v in grads.items()], dim=1)

        pair_grad = grads[::2] - grads[1::2]

        for i, sign in enumerate(signs):
            if sign:
                self.next_orders[self.left] = self.orders[self.idx]
                self.idx += 1
                self.next_orders[self.right] = self.orders[self.idx]
                self.acc += pair_grad[i]
            else:
                self.next_orders[self.right] = self.orders[self.idx]
                self.idx += 1
                self.next_orders[self.left] = self.orders[self.idx]
                self.acc -= pair_grad[i]

            self.idx += 1
            self.left += 1
            self.right -= 1

    def reset(self):
        assert self.left > self.right
        assert self.idx == self.n

        self.idx = 0
        self.orders.copy_(self.next_orders)
        self.next_orders.zero_()

        self.left = 0
        self.right = self.n - 1

        self.acc.zero_()

        # print(self.orders[:128])
        # print(self.orders[-128:])

    def __len__(self):
        return self.n

    def __iter__(self):
        yield from self.orders


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
    adaptive: bool = field(
        default=False,
        metadata={
            "help": "Use adaptive SAM",
        },
    )
    rho: float = field(
        default=0.5,
        metadata={
            "help": "Rho for SAM",
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


@torch.no_grad()
def train(
    train_loader: DataLoader,
    sampler: DESampler,
    params: dict[str, nn.Parameter],
    buffers: dict[str, Tensor],
    ft_compute_sample_grad_and_loss: callable,
    optimizer: GradientTransformation,
    opt_state: dict[str, Tensor],
    metric: evaluate.EvaluationModule,
    pbar: tqdm,
    device: torch.device = torch.device("cuda"),
    adaptive: bool = False,
    rho: float = 0.05,
) -> tuple[float, dict[str, nn.Parameter], dict[str, Tensor]]:
    sampler.reset()
    running_loss, n = 0, 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # First run of computing the gradients
        ft_per_sample_grads, (batch_loss, logits) = ft_compute_sample_grad_and_loss(
            params, buffers, x, y
        )

        # Gary: First compute signs of grads
        signs = sampler.compute_sings(ft_per_sample_grads)

        grads = {k: g.mean(dim=0) for k, g in ft_per_sample_grads.items()}

        # Sharpness-aware minimization
        norm = torch.stack(
            [
                # https://github.com/davda54/sam/issues/16
                (params[k] * g if adaptive else g).norm(p=2)
                for k, g in grads.items()
            ]
        ).norm(p=2)
        scale = rho / (norm + 1e-12)
        old_params = {k: p.data.clone() for k, p in params.items()}
        for k, p in params.items():
            if k not in grads:
                continue
            e_w = torch.pow(p, 2) * grads[k] * scale if adaptive else grads[k] * scale
            p.add_(e_w)

        # updates, opt_state = optimizer.update(
        #     grads, opt_state, params=params
        # )  # get updates
        # params = torchopt.apply_updates(params, updates)  # update network parameters

        running_loss += batch_loss.sum().item()
        n += len(batch_loss)
        metric.add_batch(predictions=logits.argmax(dim=-1), references=y)

        # Just an extra run without sampler step
        ft_per_sample_grads, (batch_loss, logits) = ft_compute_sample_grad_and_loss(
            params, buffers, x, y
        )

        # Then update acc and orders according to the signs
        sampler.step(ft_per_sample_grads, signs)

        grads = {k: g.mean(dim=0) for k, g in ft_per_sample_grads.items()}

        # Sharpness-aware minimization
        for k, p in params.items():
            if k not in grads:
                continue
            p.data = old_params[k]

        updates, opt_state = optimizer.update(
            grads, opt_state, params=params
        )  # get updates
        params = torchopt.apply_updates(params, updates)  # update network parameters

        running_loss += batch_loss.sum().item()
        n += len(batch_loss)
        metric.add_batch(predictions=logits.argmax(dim=-1), references=y)

        pbar.update(1)

    return running_loss / n / 2, params, opt_state


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
        "exp": "asam" if args.adaptive else "sam",
    }
    wandb.init(
        project=f"grab-{args.dataset_name}"
        if train_args.wandb_project is None
        else train_args.wandb_project,
        name=f"sam-{args.model_name}-da_{args.data_augmentation}-func",
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
    # orders = None
    # if grab_args.order_path is not None:
    #     orders = torch.load(grab_args.order_path)
    #     if len(orders.shape) == 2:
    #         orders = orders[-1].tolist()
    #     else:
    #         orders = orders.tolist()

    # sampler = GraBSampler(
    #     train_dataset,
    #     params,
    #     # Probabilistic balance
    #     orders=orders,
    #     # Other specific
    #     timer=timer,
    #     record_herding=grab_args.report_grads,
    #     stale_mean_herding=False,
    #     cuda_herding=not grab_args.cpu_herding,
    #     record_norm=grab_args.report_grads,
    #     # For NTK
    #     model=model,
    #     params=params,
    #     buffers=buffers,
    #     loss_fn=loss_fn,
    #     dataset=train_dataset,
    #     kernel_dtype=torch.float32,
    #     **config,
    # )
    sampler = DESampler(
        len(train_dataset),
        d,
    )

    # Initiate data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args.train_batch_size,
        # Support for RR
        sampler=None
        if grab_args.balance_type == BalanceType.RANDOM_RESHUFFLING
        else sampler,
        # Support for RR
        shuffle=True
        if grab_args.balance_type == BalanceType.RANDOM_RESHUFFLING
        else False,
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

    optimizer, _ = get_optimizer(
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
    # Now we are running data echoing
    for epoch in range(0 if train_args.log_first_step else 1, train_args.epochs + 1):
        logs = {
            "epoch": epoch,
            "iteration": epoch * len(train_loader),
        }

        if epoch != 0:
            with timer("train"):
                model.train()
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
                    adaptive=args.adaptive,
                    rho=args.rho,
                )
            # train_acc = train_metric.compute()["accuracy"]
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
            model.eval()
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
