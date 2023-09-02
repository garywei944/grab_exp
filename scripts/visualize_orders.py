import torch

import click
from absl import logging
import matplotlib.pyplot as plt


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("-d", "--max-d", default=5000)
def cli(
    file_path,
    max_d,
):
    logging.set_verbosity(logging.INFO)
    orders = torch.load(file_path)

    n, d = orders.shape

    if d > max_d:
        kernel_size = max(2, d // max_d)
        logging.info(f"d = {d} is too large to visualize. Blur and subsample.")
        blurrer = torch.nn.AvgPool1d(kernel_size=kernel_size)
        orders = blurrer(orders)

    logging.info(f"orders.shape = {orders.shape}")
    plt.figure(figsize=(16, 16), dpi=300)
    plt.imshow(orders, aspect="auto", interpolation="nearest")
    plt.show()
    plt.savefig(f"{file_path}.png")


if __name__ == "__main__":
    cli()
