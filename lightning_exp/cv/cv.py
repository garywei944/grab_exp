import torch
from torch import nn, Tensor
import numpy as np

import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from torchmetrics import Accuracy

from lightning.pytorch.demos.boring_classes import BoringDataModule

import torchopt

from cd2root import cd2root

cd2root()

from lightning_exp.cv.datamodules import CVDataModule, CIFAR10DataModule

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--demo", action="store_true", help="Run demo")

        parser.link_arguments("data", "model.dm", apply_on="instantiate")


class Model(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        dm: CVDataModule,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        adam_beta1: float,
        adam_beta2: float,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        # init models
        self.model = nn.Linear(np.prod(dm.dims), dm.num_classes)

        # convert it to functorch
        self.params = dict(self.model.named_parameters())
        self.buffers = dict(self.model.named_buffers())
        self.params = {k: v for k, v in self.params.items() if v.requires_grad}
        for v in self.params.values():
            v.requires_grad_(False)

        self.loss_fn = nn.CrossEntropyLoss()

        # init grab sampler
        self.sampler = ...

        # init optimizer
        self.optimizer = self.get_optimizer()
        self.opt_state = self.optimizer.init(self.params)

        # init metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=dm.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=dm.num_classes)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        ft_per_sample_grads, (batch_loss, logits) = ft_compute_sample_grad_and_loss(
            self.params, self.buffers, x, y
        )
        self.sampler.step(ft_per_sample_grads)

        grads = {k: g.mean(dim=0) for k, g in ft_per_sample_grads.items()}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, params=self.params, inplace=True
        )  # get updates
        self.params = torchopt.apply_updates(
            self.params, updates, inplace=True
        )  # update network parameters

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def get_optimizer(self):
        # Initiate optimizer
        if self.hparams.optimizer == "sgd":
            optimizer = torchopt.sgd(
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer in ["adam", "adamw"]:
            optimizer = torchopt.adamw(
                self.hparams.learning_rate,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                weight_decay=self.hparams.weight_decay,
                use_accelerated_op=True,
            )
        else:
            raise ValueError("Unknown optimizer")

        return optimizer


def main():
    cli = CLI(Model, CVDataModule, subclass_mode_data=True, run=False)

    print(cli.parser)
    # print(cli.parser.parse_args())
    # print(cli.parser.demo)
    print(cli.config)


if __name__ == "__main__":
    main()
