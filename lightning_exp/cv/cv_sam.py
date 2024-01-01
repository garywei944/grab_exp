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

from datamodules import CVDataModule, CIFAR10DataModule
from cv import Model

from cd2root import cd2root

cd2root()

from experiments.cv.models import WRN
from experiments.sam import SAM, enable_running_stats, disable_running_stats


class SAMModel(Model):
    def __init__(
        self,
        model_name: str = "wrn",
        dims: tuple[int, ...] = (3, 32, 32),
        num_classes: int = 10,
        optimizer: str = "sgd",
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        rho: float = 0.05,
    ):
        super().__init__(
            model_name=model_name,
            dims=dims,
            num_classes=num_classes,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
        )
        self.rho = rho

        self.save_hyperparameters()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch

        def closure():
            # Second back-prop
            disable_running_stats(self)
            loss = self.loss_fn(self(x), y)
            self.manual_backward(loss)
            self.clip_gradients(
                opt, gradient_clip_val=5.0, gradient_clip_algorithm="norm"
            )
            return loss

        # First back-prop
        enable_running_stats(self)
        loss = self.loss_fn(self(x), y)
        self.manual_backward(loss)

        opt.step(closure=closure)
        opt.zero_grad()

        sch = self.lr_schedulers()
        sch.step()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [
        #             p
        #             for n, p in self.named_parameters()
        #             if all(nd not in n for nd in no_decay)
        #         ],
        #         "weight_decay": self.weight_decay,
        #     },
        #     {
        #         "params": [
        #             p
        #             for n, p in self.named_parameters()
        #             if any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": 0.0,
        #     },
        # ]
        optimizer_grouped_parameters = self.parameters()
        if self.optimizer == "sgd":
            optimizer = SAM(
                optimizer_grouped_parameters,
                torch.optim.SGD,
                rho=self.rho,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=True,
            )
        elif self.optimizer == "adam":
            optimizer = SAM(
                optimizer_grouped_parameters,
                torch.optim.AdamW,
                rho=self.rho,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        else:
            raise ValueError

        # Use cosine scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


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

    parser.add_lightning_class_args(SAMModel, "model")
    parser.add_lightning_class_args(CIFAR10DataModule, "data")
    # parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
    # parser.link_arguments("model.task_name", "data.task_name")
    # parser.link_arguments("data", "model.dm", apply_on="instantiate")
    # parser.link_arguments("model.dm", "data", apply_on="instantiate")

    return parser.parse_args()


def main():
    timestamp = datetime.now().isoformat()
    args = parse_args()

    model_name = f"sam-{args.model.model_name}-da_{args.data.data_augmentation}"
    # task_name = args.model.task_name

    L.seed_everything(args.seed)

    dm = CIFAR10DataModule(**vars(args.data))
    model = SAMModel(**vars(args.model))

    trainer = L.Trainer(
        # max_steps=args.max_steps,
        max_epochs=args.epochs,
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
