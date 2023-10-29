import torch

from pathlib import Path
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from grabsampler import BalanceType
from grabsampler.utils.random_projection import RandomProjectionType


@dataclass
class GraBArgs:
    balance_type: BalanceType = field(
        default=BalanceType.MEAN_BALANCE,
        metadata={
            "aliases": ["-bt"],
            "choices": BalanceType,
            "help": "Balance type for GraBSampler",
        },
    )
    random_first_epoch: bool = field(
        default=True,
        metadata={
            "aliases": ["-rfe"],
            "help": "Whether to use random reshuffling the first epoch",
        },
    )
    prob_balance: bool = field(
        default=False,
        metadata={"aliases": ["-prob"], "help": "Whether to use probabilistic balance"},
    )
    prob_balance_c: float = field(
        default=30,
        metadata={"aliases": ["-c"], "help": "Coefficient of probabilistic balance"},
    )
    depth: int = field(default=5, metadata={"help": "Recursive depth of GraB"})
    normalize_grad: bool = field(
        default=False,
        metadata={
            "aliases": ["-norm"],
            "help": "Whether to normalize gradients before GraB",
        },
    )
    random_projection: RandomProjectionType = field(
        default=RandomProjectionType.NONE,
        metadata={
            "aliases": ["-rp"],
            "choices": RandomProjectionType,
            "help": "Whether to use random projection before GraB, i.e. balance PI@g "
            "instead of g",
        },
    )
    random_projection_eps: float = field(
        default=0.1, metadata={"aliases": ["-rpe"], "help": "Epsilon of JL Lemma"}
    )
    kron_order: int = field(
        default=2,
        metadata={
            "aliases": ["-ko"],
            "help": "Order of Kronecker product of random project matrices, i.e. "
            "number of element matrices to use to construct the final projection matrix",
        },
    )
    ema_decay: float = field(
        default=0.1, metadata={"help": "Decay rate of exponential moving average"}
    )
    order_path: Path = field(
        default=None,
        metadata={
            "help": "Path to the orders pt file, only used by FixedOrdering",
        },
    )
    record_orders: bool = field(
        default=False, metadata={"aliases": ["-ro"], "help": "Whether to record orders"}
    )
    report_grads: bool = field(
        default=False,
        metadata={
            "aliases": ["-rg"],
            "help": "Whether to report norms, herding, and average gradient errors",
        },
    )
    cpu_herding: bool = field(
        default=False,
        metadata={"aliases": ["-cpu_herding"], "help": "Whether to use CPU herding"},
    )

    def __post_init__(self):
        self.balance_type = BalanceType(self.balance_type)
        self.random_projection = RandomProjectionType(self.random_projection)


@dataclass
class TrainArgs:
    optimizer: str = field(
        default="adam",
        metadata={
            "aliases": ["-opt"],
            "choices": ["adam", "adamw", "sgd"],
            "help": "Optimizer",
        },
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={
            "aliases": ["-lr"],
            "help": "Learning rate",
        },
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={
            "aliases": ["-b1"],
            "help": "Adam beta1 / momentum",
        },
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={
            "aliases": ["-b2"],
            "help": "Adam beta2",
        },
    )
    weight_decay: float = field(
        default=0.01,
        metadata={
            "aliases": ["-wd"],
            "help": "Weight decay",
        },
    )
    train_batch_size: int = field(
        default=16,
        metadata={
            "aliases": ["-b"],
            "help": "Batch size",
        },
    )
    eval_batch_size: int = field(
        default=64,
        metadata={
            "aliases": ["-eb"],
            "help": "Evaluation batch size",
        },
    )
    epochs: int = field(
        default=100,
        metadata={
            "aliases": ["-e"],
            "help": "Number of epochs",
        },
    )
    max_iter: int = field(
        default=int(1e6),
        metadata={
            "aliases": ["-T"],
            "help": "Maximum number of iterations",
        },
    )
    num_workers: int = field(
        default=1,
        metadata={
            "aliases": ["-nw"],
            "help": "Number of workers for data loading",
        },
    )
    log_freq: int = field(
        default=100,
        metadata={
            "aliases": ["-lf"],
            "help": "Logging frequency",
        },
    )
    log_first_step: bool = field(
        default=False,
        metadata={
            "aliases": ["-lfs"],
            "help": "Log first step before training",
        },
    )
    checkpoint: Path = field(
        default=None,
        metadata={
            "aliases": ["-ckpt"],
            "help": "Path to checkpoint",
        },
    )
    save_steps: int = field(
        default=int(1e4),
        metadata={
            "help": "Save steps",
        },
    )
    save_strategy: str = field(
        default="steps",
        metadata={
            "aliases": ["-ss"],
            "choices": ["no", "epoch", "steps"],
            "help": "Save strategy",
        },
    )
    seed: int = field(
        default=42,
        metadata={
            "aliases": ["-s"],
            "help": "Random seed",
        },
    )
    wandb: bool = field(
        default=False,
        metadata={
            "help": "Use wandb for logging",
        },
    )
    wandb_project: str = field(
        default=None,
        metadata={
            "aliases": ["-wp"],
            "help": "Wandb project name",
        },
    )
    tqdm: bool = field(
        default=True,
        metadata={
            "help": "Disable tqdm progress bar",
        },
    )
    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        metadata={
            "help": "PyTorch Device",
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Use mixed precision training",
        },
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": "Use mixed precision training",
        },
    )
    output_path: Path = field(
        default=Path("checkpoints"),
        metadata={
            "aliases": ["-op"],
            "help": "Output path",
        },
    )

    def __post_init__(self):
        if self.bf16:
            self.dtype = torch.bfloat16
        elif self.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32


if __name__ == "__main__":
    args: GraBArgs
    train_args: TrainArgs
    (args, train_args) = HfArgumentParser(
        (GraBArgs, TrainArgs)
    ).parse_args_into_dataclasses()

    print(args)
    print(train_args.device)
    print(type(train_args.device))
