from typing import Any

import torch
from torch import nn, Tensor
from torch.func import functional_call, grad_and_value, vmap

import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    Timer,
    LearningRateMonitor,
)
from torchmetrics import Accuracy
import torchopt

from datetime import datetime
from functools import partial

from datamodules import CVDataModule, CIFAR10DataModule

from cd2root import cd2root

cd2root()

from experiments.cv.models import WRN


def multi_step_lr(
    learning_rate: float,
    milestones: list[int],
    gamma: float = 0.1,
):
    def _multi_step_lr(step: int):
        return learning_rate * gamma ** sum(step > m for m in milestones)

    return _multi_step_lr


def compute_loss(model, loss_fn, params, buffers, inputs, targets):
    inputs = inputs.unsqueeze(0)
    targets = targets.unsqueeze(0)

    logits = functional_call(model, (params, buffers), (inputs,))

    return loss_fn(logits, targets), logits


class Model(L.LightningModule):
    def __init__(
        self,
        dm: CVDataModule = None,
        model_name: str = "wrn",
        optimizer: str = "sgd",
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        milestones: list[int] = None,
        norm: str = "bn",
    ):
        super().__init__()

        self.dims = dm.dims
        self.num_classes = dm.num_classes

        self.model_name = model_name
        self.opt = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.milestones = milestones
        self.norm = norm

        self.save_hyperparameters(ignore="dm")

        if self.model_name == "wrn":
            self.model = WRN(n_classes=self.num_classes, norm=self.norm)
        elif self.model_name == "resnet":
            if self.norm == "bn":
                from torchvision.models import resnet18

                self.model = resnet18(num_classes=self.num_classes)
            elif self.norm == "gn":
                from experiments.cv.models import resnet18

                self.model = resnet18(num_classes=self.num_classes)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.fparams = dict(self.model.named_parameters())
        self.fbuffers = dict(self.model.named_buffers())
        self.fparams = {k: v for k, v in self.fparams.items() if v.requires_grad}
        for v in self.fparams.values():
            v.requires_grad_(False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = Accuracy(task="multiclass", num_classes=self.num_classes)

        self.automatic_optimization = False

        self.ft_grad_loss = vmap(
            grad_and_value(
                partial(compute_loss, self.model, self.loss_fn), has_aux=True
            ),
            in_dims=(None, None, 0, 0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @torch.no_grad()
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        per_sample_grads, (batch_loss, logits) = self.ft_grad_loss(
            self.fparams,
            self.fbuffers,
            *batch,
        )

        grads = {k: v.mean(dim=0) for k, v in per_sample_grads.items()}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, params=self.fparams
        )
        self.fparams = torchopt.apply_updates(self.fparams, updates)

        self.log(
            "train_loss",
            batch_loss.mean(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        # self.metrics(logits, y)
        self.metrics.update(logits, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("val_acc", self.metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def func_optimizers(self):
        if self.opt == "sgd":
            optimizer = torchopt.sgd(
                lr=multi_step_lr(
                    learning_rate=self.learning_rate,
                    milestones=[60, 120, 160],
                    gamma=0.2,
                ),
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        elif self.opt == "adam":
            no_decay = ["bias", "LayerNorm.weight"]

            # mask = torchopt.transform.masked(
            #     torchopt.transform.add_decayed_weights(self.weight_decay),
            #     lambda name: all(nd not in name for nd in no_decay),
            # )
            optimizer = torchopt.adamw(
                lr=multi_step_lr(
                    learning_rate=self.learning_rate,
                    milestones=self.milestones,
                    gamma=0.2,
                ),
                betas=(self.adam_beta1, self.adam_beta2),
                weight_decay=self.weight_decay,
                mask={
                    k: all(nd not in k for nd in no_decay) for k in self.fparams.keys()
                },
            )
        else:
            raise NotImplementedError

        optimizer = torchopt.chain(torchopt.clip_grad_norm(5.0), optimizer)

        return optimizer

    def configure_optimizers(self):
        # torchopt will be passed to correct device
        self.optimizer = self.func_optimizers()
        self.opt_state = self.optimizer.init(self.fparams)

        # Dummy optimizer
        return torch.optim.SGD(self.parameters(), lr=0.0)


def parse_args():
    parser = LightningArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_lightning_class_args(Model, "model")
    parser.add_lightning_class_args(CIFAR10DataModule, "data")

    args = parser.parse_args()
    del args.model.dm, args.model.milestones

    return args


def main():
    timestamp = datetime.now().isoformat()
    args = parse_args()

    model_name = f"{args.model.model_name}-da_{args.data.data_augmentation}"

    L.seed_everything(args.seed)

    dm = CIFAR10DataModule(**vars(args.data))
    dm.prepare_data()
    dm.setup()

    n_batches = len(dm.train_dataloader())
    model = Model(
        dm=dm,
        milestones=[60 * n_batches, 120 * n_batches, 160 * n_batches],
        **vars(args.model),
    )

    trainer = L.Trainer(
        # max_steps=args.max_steps,
        # max_epochs=args.epochs,
        max_steps=args.epochs * len(dm.train_dataloader()),
        check_val_every_n_epoch=1,
        # val_check_interval=args.val_interval,
        # gradient_clip_val=5.0,
        logger=[
            TensorBoardLogger("lightning_logs", name=model_name),
            # WandbLogger(
            #     project=f"sam-cifar10",
            #     name=model_name,
            #     entity="grab",
            #     mode="online",
            # ),
            CSVLogger("logs", name=model_name),
        ],
        callbacks=[
            # EarlyStopping(monitor="val_loss"),
            # ModelCheckpoint(
            #     # monitor="matthews_correlation",
            #     save_top_k=-1,
            #     dirpath=f"checkpoints/{model_name}/glue/{timestamp}",
            #     every_n_train_steps=1000,
            #     # save_weights_only=True,
            #     save_last=True,
            #     verbose=True,
            # ),
            # LearningRateMonitor(logging_interval="step"),
            Timer(),
        ],
        # profiler="simple",
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        # limit_val_batches=0,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    # trainer.validate(model, datamodule=dm)
    trainer.fit(
        model,
        datamodule=dm,
        # ckpt_path="lightning_logs/wrn/version_6/checkpoints/epoch=39-step=7840.ckpt",
    )


if __name__ == "__main__":
    main()
