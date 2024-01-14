from cvxpy import logistic
import torch
from torch import nn, Tensor
import numpy as np

import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    Timer,
    LearningRateMonitor,
)
from torchmetrics import Accuracy

from lightning.pytorch.demos.boring_classes import BoringDataModule

from transformers import get_cosine_schedule_with_warmup

from datetime import datetime

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from grabsampler import GraBSampler

from datamodules import CVDataModule, CIFAR10DataModule
from cv import Model

from cd2root import cd2root

cd2root()

from lightning.pytorch.trainer.trainer import log

log.setLevel("DEBUG")


class BackpackModel(Model):
    def __init__(
        self,
        dm: CVDataModule = None,
        model_name: str = "resnet",
        optimizer: str = "sgd",
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        rho: float = 0.05,
        norm: str = "bn",
        gradient_clip_val: float = None,
    ):
        super().__init__(
            dm=dm,
            model_name=model_name,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            norm=norm,
        )
        self.rho = rho
        self.gradient_clip_val = gradient_clip_val

        self.save_hyperparameters(ignore="dm")

        self.dm = dm

        self.model = extend(self.model)
        self.loss_fn = extend(self.loss_fn)

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch

        b = x.shape[0]

        # First back-prop
        loss = self.loss_fn(self(x), y)

        with backpack(BatchGrad()):
            self.manual_backward(loss)

        if self.gradient_clip_val:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        opt.step()
        opt.zero_grad()

        # sch = self.lr_schedulers()
        # sch.step()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        per_sample_grad = torch.cat(
            [
                p.grad_batch.reshape(b, -1)
                for p in self.parameters()
                if p.grad_batch is not None
            ],
            dim=1,
        )

        self.dm.sampler.step({"": per_sample_grad})

        return loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()


def parse_args():
    parser = LightningArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    # parser.add_argument("-T", "--max_steps", type=int, default=2500)
    # parser.add_argument(
    #     "-vi",
    #     "--val_interval",
    #     type=int,
    #     default=100,
    # )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sam-cifar10",
    )
    parser.add_argument("-bt", "--balance", choices=["rr", "mean"], default="mean")

    parser.add_lightning_class_args(BackpackModel, "model")
    parser.add_lightning_class_args(CIFAR10DataModule, "data")
    # parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
    # parser.link_arguments("model.task_name", "data.task_name")
    # parser.link_arguments("data", "model.dm", apply_on="instantiate")
    # parser.link_arguments("model.dm", "data", apply_on="instantiate")

    args = parser.parse_args()
    del args.model.dm

    return args


def main():
    timestamp = datetime.now().isoformat()
    args = parse_args()

    model_name = f"{args.model.model_name}-da_{args.data.data_augmentation}-{args.model.norm}-backpack-{args.balance}"
    # task_name = args.model.task_name

    L.seed_everything(args.seed)

    dm = CIFAR10DataModule(**vars(args.data))
    dm.prepare_data()
    dm.setup()
    model = BackpackModel(dm=dm, **vars(args.model))

    sampler = GraBSampler(
        dm.train_dataloader(), dict(model.named_parameters()), args.balance
    )
    dm.sampler = sampler

    trainer = L.Trainer(
        # max_steps=args.max_steps,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        # val_check_interval=args.val_interval,
        # gradient_clip_val=5.0,
        logger=[
            # TensorBoardLogger("lightning_logs", name=model_name),
            WandbLogger(
                project=args.wandb_project,
                name=model_name,
                entity="grab",
                mode="online",
            ),
            # CSVLogger("logs", name=model_name),
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
            LearningRateMonitor(logging_interval="step"),
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
        # ckpt_path="checkpoints/google/t5-v1_1-small/glue/2023-12-28T18:12:26.650635/epoch=7-step=54000.ckpt",
    )


if __name__ == "__main__":
    main()
