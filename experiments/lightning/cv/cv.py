from ast import alias
import torch
from torch import nn, Tensor

import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    Timer,
    LearningRateMonitor,
)
from torchmetrics import Accuracy

from datetime import datetime
from functools import partial

from datamodules import CVDataModule, CIFAR10DataModule

from cd2root import cd2root

cd2root()


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
        norm: str = "bn",
    ):
        super().__init__()

        self.dims = dm.dims
        self.num_classes = dm.num_classes

        self.model_name = model_name
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.norm = norm

        self.save_hyperparameters()

        if self.model_name == "wrn":
            from experiments.cv.models import WRN

            self.model = WRN(n_classes=self.num_classes, norm=self.norm)
        elif self.model_name == "resnet":
            if self.norm == "bn":
                from experiments.cv.models.preact_resnet import PreActResNet18

                self.model = PreActResNet18(n_cls=self.num_classes)
            elif self.norm == "gn":
                from experiments.cv.models.preact_resnet import PreActResNet18GroupNorm

                self.model = PreActResNet18GroupNorm(n_cls=self.num_classes)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

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
        self.log("val_acc", self.metrics, on_step=False, on_epoch=True)

        return loss

    # def on_validation_epoch_end(self) -> None:
    #     self.log("val_acc", self.metrics.compute())
    #     self.metrics.reset()

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
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=True,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        else:
            raise ValueError

        # Use cosine scheduler
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=self.trainer.max_steps,
        # )
        if self.model_name == "wrn":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[60, 120, 160],
                gamma=0.2,
            )
        elif self.model_name == "resnet":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[100, 150],
                gamma=0.1,
            )
        else:
            raise NotImplementedError

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def parse_args():
    parser = LightningArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sam-cifar10",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
    )

    parser.add_lightning_class_args(Model, "model")
    parser.add_lightning_class_args(CIFAR10DataModule, "data")

    args = parser.parse_args()
    del args.model.dm

    return args


def main():
    timestamp = datetime.now().isoformat()
    args = parse_args()

    model_name = (
        f"{args.model.model_name}-da_{args.data.data_augmentation}-{args.model.norm}"
    )

    L.seed_everything(args.seed)

    dm = CIFAR10DataModule(**vars(args.data))
    model = Model(dm=dm, **vars(args.model))

    trainer = L.Trainer(
        # max_steps=args.max_steps,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        # val_check_interval=args.val_interval,
        gradient_clip_val=args.grad_clip,
        logger=[
            TensorBoardLogger("lightning_logs", name=model_name),
            WandbLogger(
                project=args.wandb_project,
                name=model_name,
                entity="grab",
                mode="online",
            ),
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
        enable_checkpointing=True,
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
        ckpt_path=args.ckpt,
    )


if __name__ == "__main__":
    main()
