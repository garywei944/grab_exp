from dataclasses import dataclass, field
from grablib import BalanceType
from grablib.utils.random_projection import RandomProjectionType


@dataclass
class GraBArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # GraB specific arguments
    balance_type: str = field(
        default=BalanceType.MEAN_BALANCE.value,
        metadata={
            "help": "Balance type for GraBSampler",
            "choices": [e.value for e in BalanceType],
        },
    )
    depth: int = field(default=5, metadata={"help": "Recursive depth of GraB"})
    normalize_grads: bool = field(
        default=False, metadata={"help": "Whether to normalize gradients before GraB"}
    )
    random_projection: str = field(
        default=RandomProjectionType.NONE.value,
        metadata={
            "help": "Whether to use random projection before GraB, i.e. balance PI@g "
            "instead of g",
            "choices": [e.value for e in RandomProjectionType],
        },
    )
    random_projection_eps: float = field(
        default=0.1, metadata={"help": "Epsilon of JL Lemma"}
    )
    kron_order: int = field(
        default=2,
        metadata={
            "help": "Order of Kronecker product of random project matrices, i.e. "
            "number of element matrices to use to construct the final projection matrix"
        },
    )
    prob_balance: bool = field(
        default=False, metadata={"help": "Whether to use probabilistic balance"}
    )
    prob_balance_c: float = field(
        default=30, metadata={"help": "Coefficient of probabilistic balance"}
    )
    random_first_epoch: bool = field(
        default=True,
        metadata={"help": "Whether to use random reshuffling the first epoch"},
    )

    # EMA Balance
    ema_decay: float = field(
        default=0.1, metadata={"help": "Decay rate of exponential moving average"}
    )

    # Fixed Order
    order_path: str = field(
        default=None,
        metadata={"help": "Path to the orders pt file, only used by FixedOrdering"},
    )

    # NTK based
    centered_feature_map: bool = field(
        default=False,
        metadata={
            "help": "Whether to substitute the per-sample gradient with its mean"
        },
    )
    largest_eig: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the largest eigenvalue of the Kernel to find an order"
        },
    )
    descending_eig: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the descending order of the eigenvector to find an order"
        },
    )
    abs_eig: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the absolute value of the eigenvector to find an order"
        },
    )
    save_k: bool = field(
        default=False,
        metadata={
            "help": "Whether to save the kernel matrix during training, only for research purposes."
        },
    )
    kernel_device: str = field(
        default="cuda",
        metadata={
            "help": "Device to store the kernel matrix of size (n, n)",
        },
    )

    # Order by Norm
    descending_norm: bool = field(
        default=True,
        metadata={
            "help": "Whether to use the descending order of the norm to find an order"
        },
    )

    # Experiment specific arguments
    record_orders: bool = field(
        default=True, metadata={"help": "Whether to record orders"}
    )
    record_grads: bool = field(
        default=False,
        metadata={
            "help": "Whether to record norms, herding, and average gradient errors"
        },
    )
    cpu_herding: bool = field(
        default=False, metadata={"help": "Whether to use CPU herding"}
    )
    use_wandb: bool = field(default=False, metadata={"help": "Whether to use wandb"})
    wandb_project: str = field(
        default=None,
        metadata={"help": "Wandb project name"},
    )
