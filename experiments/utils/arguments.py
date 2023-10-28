import torch

from typing import Literal
from tap import Tap
from pathlib import Path

from grabsampler import BalanceType
from grabsampler.utils.random_projection import RandomProjectionType


class GraBArguments(Tap):
    balance_type: BalanceType = (
        BalanceType.MEAN_BALANCE
    )  # Balance type for GraB-Sampler
    random_first_epoch: bool = True  # Whether to use random reshuffling the first epoch
    prob_balance: bool = False  # Whether to use probabilistic balance
    prob_balance_c: float = 30  # Coefficient of probabilistic balance

    depth: int = 5  # Depth of Recursive Balance
    normalize_grad: bool = False  # Whether to normalize gradients before GraB
    random_projection: RandomProjectionType = (
        RandomProjectionType.NONE
    )  # Whether to use random projection before GraB, i.e. balance PI@g
    random_projection_eps: float = 0.1  # Epsilon of JL Lemma
    kron_order: int = 2
    """
    Order of Kronecker product of random project matrices, i.e. number of element
    matrices to use to construct the final projection matrix
    """

    # EMA Balance
    ema_decay: float = 0.1  # Decay rate of exponential moving average

    # Fixed Order
    order_path: Path = None  # Path to the orders pt file, only used by FixedOrdering

    # Experiment specific arguments
    record_orders: bool = False  # Whether to record the orders
    report_grads: bool = False
    "Whether to report norms, herding, and average gradient errors"
    cpu_herding: bool = False  # Whether to use CPU herding

    def configure(self) -> None:
        self.add_argument("-bt", "--balance_type")
        self.add_argument("-rfe", "--random_first_epoch")
        self.add_argument("-prob", "--prob_balance")
        self.add_argument("-c", "--prob_balance_c")
        self.add_argument("-norm", "--normalize_grad")
        self.add_argument("-rp", "--random_projection")
        self.add_argument("-rpe", "--random_projection_eps")
        self.add_argument("-ko", "--kron_order")
        self.add_argument("-ro", "--record_orders")
        self.add_argument("-rg", "--report_grads")


class TrainArgs(Tap):
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"  # Optimizer
    learning_rate: float = 1e-3  # Learning rate
    adam_beta1: float = 0.9  # Adam beta1 / momentum
    adam_beta2: float = 0.999  # Adam beta2
    weight_decay: float = 0.01  # Weight decay
    batch_size: int = 16  # Batch size
    eval_batch_size: int = 64  # Evaluation batch size
    epochs: int = 100  # Number of epochs
    max_iter: int = int(1e6)  # Maximum number of iterations
    num_workers: int = 1  # Number of workers for data loading
    log_freq: int = 100  # Logging frequency
    log_first_step: bool = False  # Log first step before training
    checkpoint: Path = None  # Path to checkpoint
    save_steps: int = int(1e4)  # Save steps
    save_strategy: Literal["no", "epoch", "steps"] = "steps"  # Save strategy
    seed: int = 42  # Random seed
    wandb: bool = False  # Use wandb for logging
    wandb_project: str = None  # Wandb project name
    tqdm: bool = True  # Disable tqdm progress bar
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # PyTorch Device
    fp16: bool = False  # Use mixed precision training
    bf16: bool = False
    output_path: Path = Path("checkpoints")

    def configure(self) -> None:
        self.add_argument("-opt", "--optimizer")
        self.add_argument("-lr", "--learning_rate")
        self.add_argument("-b1", "--adam_beta1")
        self.add_argument("-b2", "--adam_beta2")
        self.add_argument("-wd", "--weight_decay")
        self.add_argument("-b", "--batch_size")
        self.add_argument("-eb", "--eval_batch_size")
        self.add_argument("-e", "--epochs")
        self.add_argument("-T", "--max_iter")
        self.add_argument("-nw", "--num_workers")
        self.add_argument("-lf", "--log_freq")
        self.add_argument("-lfs", "--log_first_step")
        self.add_argument("-ckpt", "--checkpoint")
        self.add_argument("-s", "--seed")
        self.add_argument("-wp", "--wandb_project")
        self.add_argument("-op", "--output_path")

    def process_args(self) -> None:
        if self.bf16:
            self.dtype = torch.bfloat16
        elif self.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
