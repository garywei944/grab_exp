import os
import sys
from pathlib import Path

import torch
from sklearn.random_projection import (
    johnson_lindenstrauss_min_dim,
    GaussianRandomProjection,
    SparseRandomProjection,
)
import torch_sparse
from torch_sparse import SparseTensor
from torch_sparse.tensor import from_scipy
from accelerate.utils import set_seed

import numpy as np
import pandas as pd
import click
import wandb
from absl import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Change pwd to the project root directory
PROJECT_NAME = "GraB"
PROJECT_PATH = Path(__file__).resolve()
while PROJECT_PATH.name != PROJECT_NAME:
    PROJECT_PATH = PROJECT_PATH.parent
os.chdir(PROJECT_PATH)
sys.path.insert(0, str(PROJECT_PATH))


def herding_bound(vecs: torch.Tensor) -> float:
    if vecs.get_device() != -1:
        return torch.max(
            torch.norm(torch.cumsum(vecs, dim=0), p=torch.inf, dim=1)
        ).item()

    # Need a memory-efficient implementation for CPU computation
    n, d = vecs.shape
    acc = torch.zeros_like(vecs[0], device="cuda")
    best = -torch.inf
    for i in tqdm(range(n), total=n, leave=False):
        acc += vecs[i].cuda()
        best = max(best, torch.norm(acc, p=torch.inf).item())
    return best


def random_vector_with_2norm(_d, norm, dtype=torch.float16, device="cuda"):
    # Generate Uniform Random Variates with Constant Norm
    # https://stats.stackexchange.com/a/487505
    # https://stats.stackexchange.com/a/487505
    v = torch.normal(0, 1, (_d,), dtype=dtype, device=device)
    v = v * norm / torch.norm(v)
    return v.to(dtype=dtype)


def get_vecs(n, d, miu, sigma, dtype=torch.float16, device="cuda"):
    norms = torch.normal(miu, sigma, (n,))
    V = torch.stack(
        [random_vector_with_2norm(d, e, dtype=dtype, device=device) for e in norms]
    ).to(dtype=dtype, device=device)
    V -= V.mean(dim=0)

    logging.info(
        f"Vectors generated with norms mean {V.norm(dim=1).mean():.2f} "
        f"std {V.norm(dim=1).std():.2f}"
    )
    logging.info(f"V centered at {V.mean(dim=0)} with sum {V.mean(dim=0).sum()}")
    logging.info(f"V max {V.max()} min {V.min()}")

    return V


def run_reordering(
    V,
    n,
    d,
    epochs,
    balance="mean",
    batch_size=16,
    PI: SparseTensor | None = None,
    prob=False,
    delta=0.05,
    normalize=False,
    show_plt=False,
):
    logging.info(
        f"Running reordering with {balance} balance prob: {prob} "
        f"normalize: {normalize}"
    )
    d_proc = d
    if PI is not None:
        dd = PI.size(0)
        logging.debug("Random projection")
        *index, value = PI.coo()

        d_proc = dd

    c = 30 * np.log(n * d_proc / delta)

    if balance == "rr":
        from grablib.sorter import RandomReshuffling

        sorter = RandomReshuffling(n, d_proc)
    elif balance == "mean":
        from grablib.sorter import MeanBalance

        sorter = MeanBalance(
            n,
            d_proc,
            random_first_epoch=False,
            prob_balance=prob,
            prob_balance_c=c,
            device=torch.device("cuda"),
        )
    elif balance == "pair":
        from grablib.sorter import PairBalance

        sorter = PairBalance(
            n,
            d_proc,
            random_first_epoch=False,
            prob_balance=prob,
            prob_balance_c=c,
            device=torch.device("cuda"),
        )
    elif balance == "recursive":
        from grablib.sorter import RecursiveBalance

        sorter = RecursiveBalance(
            n,
            d_proc,
            depth=5,
            random_first_epoch=False,
            prob_balance=prob,
            prob_balance_c=c,
            device=torch.device("cuda"),
        )
    else:
        raise ValueError

    bounds = []
    orders_history = []
    logging.info(f"Init V bound: {herding_bound(V)}")

    if batch_size == 0:
        batch_size = n
    disable_tqdm = batch_size == n

    for epoch in range(epochs):
        # rr doesn't reset epoch for the first epoch
        if epoch > 0 or balance != "rr":
            sorter._reset_epoch()

        for idx in tqdm(
            range(0, n, batch_size),
            total=n // batch_size,
            leave=False,
            disable=disable_tqdm,
        ):
            V_batch = V[sorter.orders][idx : idx + batch_size, :].cuda()

            # random projection
            if PI is not None:
                # (b, dd)
                V_batch = torch_sparse.spmm(index, value, dd, d, V_batch.T).T

            # Normalize V s.t. ||v||_2 = 1
            if normalize:
                logging.debug("Normalizing V")
                # V_norm = V / torch.norm(V, p=2, dim=1).amax()
                # NOTE: vectors no more centered at 0 any more
                V_batch = torch.nn.functional.normalize(V_batch, p=2, dim=1)
                # V_batch = V_batch / NORM
                logging.debug(f"V norm: {torch.norm(V_batch, p=2, dim=1)}")

            sorter.step({"": V_batch})

        bounds.append(herding_bound(V[sorter.orders]))
        logging.info(f"Epoch {epoch}: {bounds[-1]}")
        wandb.log({"bound": bounds[-1]})
        orders_history.append(sorter.orders.tolist())
    sorter._reset_epoch()
    bounds.append(herding_bound(V[sorter.orders]))
    logging.info(f"Epoch {epochs}: {bounds[-1]}")
    wandb.log({"bound": bounds[-1]})
    orders_history.append(sorter.orders.tolist())

    orders_history = np.array(orders_history, dtype=np.uint32)

    if show_plt:
        plt.imshow(np.repeat(orders_history, n // epochs, axis=0))
        plt.show()

    return bounds


@click.command()
@click.argument("balance", type=click.Choice(["rr", "mean", "pair", "recursive"]))
@click.argument("n", type=int, default=10_000)
@click.argument("d", type=int, default=62_000)
@click.argument("miu", type=float, default=0)
@click.argument("sigma", type=float, default=1)
@click.argument("eps", type=float, default=0.1)
@click.argument("delta", type=float, default=0.5)
@click.option("-b", "--batch-size", type=int, default=0)
@click.option("-p", "--prob", is_flag=True)
@click.option("--norm", is_flag=True)
@click.option("--pi", is_flag=True)
@click.option("-D", "--dense", is_flag=True)
@click.option("-e", "--epochs", type=int, default=100)
@click.option("-s", "--seed", type=int, default=42)
@click.option("-d", "--device", type=str, default="cuda")
@click.option("-t", "--dtype", type=str, default="float16")
@click.option("-g", "--logging-level", type=str, default="info")
@click.option("-w", "--wandb", "use_wandb", is_flag=True)
def cli(
    balance,
    n,
    d,
    miu,
    sigma,
    eps,
    delta,
    batch_size,
    prob,
    norm,
    pi,
    dense,
    epochs,
    seed,
    device,
    dtype,
    logging_level,
    use_wandb,
):
    assert batch_size == 0 or n % batch_size == 0

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

    pi_id = f"{n}_{d}_{eps}_{seed}_{dtype}"

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    dd = johnson_lindenstrauss_min_dim(n, eps=eps)

    logging.info(f"Reduced dimension: {dd}")
    V = get_vecs(n, d, miu, sigma, dtype=dtype, device="cuda")
    logging.info(f"V: {V.shape}")

    # Generate PI
    file_name = Path("data/interim") / f"pi_{pi_id}.pt"
    if pi:
        if file_name.exists():
            PI = torch.load(file_name)
            PI = PI.to(dtype=dtype, device=device).cuda()
            logging.info(f"Loaded from {file_name}")

            logging.info(
                f"shape: {PI.sizes()}, nnz: {PI.nnz()}, " f"sparsity: {PI.sparsity()}"
            )
        else:
            logging.info(f"Generating {file_name}")

            if not dense:
                logging.info("Generating sparse random projection")
                transformer = SparseRandomProjection(eps=eps, random_state=seed)
            else:
                logging.info("Generating dense random projection")
                transformer = GaussianRandomProjection(eps=eps, random_state=seed)
            transformer.fit_transform(V.cpu().numpy())
            logging.info(f"Fitted transformer")

            PI = transformer.components_  # (dd, d)
            del transformer

            if not dense:
                PI: SparseTensor = from_scipy(PI).to(dtype=dtype, device=device).cuda()

                logging.info(
                    f"shape: {PI.sizes()}, nnz: {PI.nnz()}, "
                    f"sparsity: {PI.sparsity()}"
                )
                # print(PI @ SparseTensor.from_edge_index(
                #     *torch_sparse.transpose(index, value, d, dd), (dd, d),
                # ))
            else:
                PI: torch.Tensor = torch.tensor(PI, dtype=dtype, device=device)
                logging.info(f"shape: {PI.shape}")

            torch.save(PI, file_name)
            logging.info(f"Saved to {file_name}")
    else:
        PI = None

    # V = V.cuda()
    V = V.to(device=device, dtype=dtype)

    # out = torch_sparse.spmm(index, value, dd, d, V.T).T
    # print(out)
    # print(out.shape)
    #
    # print(herding_bound(out))

    # bounds = {
    #     'rr': run_reordering(
    #         V, n, d, epochs, balance='rr', prob=False
    #     ),
    #     'mean': run_reordering(
    #         V, n, d, epochs, balance='mean', prob=False
    #     ),
    #     # 'mean_prob': run_reordering(
    #     #     V, n, d, epochs, balance='mean', prob=True
    #     # ),
    #     'mean_norm': run_reordering(
    #         V, n, d, epochs, balance='mean', prob=False, normalize=True
    #     ),
    #     # 'mean_norm_prob': run_reordering(
    #     #     V, n, d, epochs, balance='mean', prob=True, normalize=True
    #     # ),
    #     'mean_pi': run_reordering(
    #         V, n, d, epochs, PI=PI, balance='mean', prob=False
    #     ),
    #     'mean_norm_pi': run_reordering(
    #         V, n, d, epochs, PI=PI, balance='mean', prob=False, normalize=True
    #     ),
    #     # 'mean_norm_prob_pi': run_reordering(
    #     #     V, n, d, epochs, PI=PI, balance='mean', prob=True, normalize=True
    #     # ),
    #     'pair': run_reordering(
    #         V, n, d, epochs, balance='pair', prob=False
    #     ),
    #     # 'pair_prob': run_reordering(
    #     #     V, n, d, epochs, balance='pair', prob=True
    #     # ),
    #     'pair_norm': run_reordering(
    #         V, n, d, epochs, balance='pair', prob=False, normalize=True
    #     ),
    #     # 'pair_norm_prob': run_reordering(
    #     #     V, n, d, epochs, balance='pair', prob=True, normalize=True
    #     # ),
    #     'pair_pi': run_reordering(
    #         V, n, d, epochs, PI=PI, balance='pair', prob=False
    #     ),
    #     'pair_norm_pi': run_reordering(
    #         V, n, d, epochs, PI=PI, balance='pair', prob=False, normalize=True
    #     ),
    #     # 'pair_norm_prob_pi': run_reordering(
    #     #     V, n, d, epochs, PI=PI, balance='pair', prob=True, normalize=True
    #     # ),
    # }
    #
    # fig = plt.gcf()
    # fig.set_size_inches(12, 9)
    # for k, v in bounds.items():
    #     plt.plot(v, label=k)
    # plt.legend()
    # plt.savefig(f'sandbox/gary/random_vector/'
    #             f'{n}_{d}_{miu}_{sigma}_{eps}_{seed}_{dtype}.png')
    # # plt.show()

    # for balance in 'mean', 'pair', 'recursive':
    name = balance
    if norm:
        name += "_norm"
    if prob:
        name += "_prob"
    if pi:
        name += "_pi"

    wandb.init(
        project="grab-random-vector",
        entity="grab",
        config={
            "balance": balance,
            "batch_size": batch_size,
            "prob": prob,
            "normalize": norm,
            "pi": pi,
            "n": n,
            "d": d,
            "miu": miu,
            "sigma": sigma,
            "eps": eps,
            "dense": dense,
            "seed": seed,
            "dtype": dtype,
        },
        name=name,
        mode="online" if use_wandb else "disabled",
    )
    bounds = run_reordering(
        V,
        n,
        d,
        epochs,
        PI=PI if pi else None,
        balance=balance,
        batch_size=batch_size,
        prob=prob,
        normalize=norm,
    )
    logging.info(f"{name} bound mean {np.mean(bounds)}")
    print(name, bounds)
    wandb.finish()


if __name__ == "__main__":
    cli()
