from cvxpy import logistic
import torch
from torch import nn, Tensor
from torch.utils.data import Sampler
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

# from grabsampler import GraBSampler

from datamodules import CVDataModule, CIFAR10DataModule
from cv import Model


from cd2root import cd2root

cd2root()

from experiments.sam import SAM, enable_running_stats, disable_running_stats
from lightning.pytorch.trainer.trainer import log

log.setLevel("DEBUG")


class SAMSampler(Sampler):
    orders: Tensor
    next_orders: Tensor
    acc: Tensor

    def __init__(self, n: int, d: int, *args, **kwargs):
        super().__init__()
        self.n = n
        self.d = d

        self.orders: Tensor = torch.randperm(n, dtype=torch.int64)
        # self.orders = torch.arange(n, dtype=torch.int64)
        self.next_orders: Tensor = self.orders.clone()

        self.acc = torch.zeros(d, dtype=torch.float32, device=torch.device("cuda"))

        self.idx = self.n
        self.left = self.n
        self.right = self.n - 1

    def compute_sings(self, grads: dict[str, Tensor]):
        b = next(iter(grads.values())).shape[0]

        assert b % 2 == 0
        acc = self.acc.clone()

        grads = torch.cat([v.reshape(b, -1) for k, v in grads.items()], dim=1)

        pair_grad = grads[::2] - grads[1::2]

        signs = []

        for i in range(b // 2):
            if torch.inner(pair_grad[i], acc) < 0:
                signs.append(True)
                acc.add_(pair_grad[i])
            else:
                signs.append(False)
                acc.sub_(pair_grad[i])

        return signs

    def step(
        self,
        grads: dict[str, Tensor],
        signs: list[bool],
    ):
        b = next(iter(grads.values())).shape[0]

        assert b % 2 == 0
        assert len(signs) == b // 2

        grads = torch.cat([v.reshape(b, -1) for k, v in grads.items()], dim=1)

        pair_grad = grads[::2] - grads[1::2]

        for i, sign in enumerate(signs):
            if sign:
                self.next_orders[self.left] = self.orders[self.idx]
                self.idx += 1
                self.next_orders[self.right] = self.orders[self.idx]
                self.acc += pair_grad[i]
            else:
                self.next_orders[self.right] = self.orders[self.idx]
                self.idx += 1
                self.next_orders[self.left] = self.orders[self.idx]
                self.acc -= pair_grad[i]

            self.idx += 1
            self.left += 1
            self.right -= 1

    def reset(self):
        assert self.left > self.right
        assert self.idx == self.n

        self.idx = 0
        self.orders.copy_(self.next_orders)
        self.next_orders.zero_()

        self.left = 0
        self.right = self.n - 1

        self.acc.zero_()

        # print(self.orders[:128])
        # print(self.orders[-128:])

    def __len__(self):
        return self.n

    def __iter__(self):
        yield from self.orders


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
        norm: str = "bn",
        gradient_clip_val: float = None,
        rho: float = 0.1,
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
        self.gradient_clip_val = gradient_clip_val
        self.rho = rho

        self.save_hyperparameters(ignore="dm")

        self.dm = dm

        self.model = extend(self.model)
        self.loss_fn = extend(self.loss_fn)

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch

        b = x.shape[0]

        def closure():
            # Second back-prop
            disable_running_stats(self)
            loss = self.loss_fn(self(x), y)

            with backpack(BatchGrad()):
                self.manual_backward(loss)

            per_sample_grad = torch.cat(
                [
                    p.grad_batch.reshape(b, -1)
                    for p in self.parameters()
                    if p.grad_batch is not None
                ],
                dim=1,
            )

            self.dm.sampler.step(
                {"": per_sample_grad},
                self.signs,
            )

            return loss

        # First back-prop
        enable_running_stats(self)
        loss = self.loss_fn(self(x), y)

        with backpack(BatchGrad()):
            self.manual_backward(loss)

        per_sample_grad = torch.cat(
            [
                p.grad_batch.reshape(b, -1)
                for p in self.parameters()
                if p.grad_batch is not None
            ],
            dim=1,
        )

        self.signs = self.dm.sampler.compute_sings({"": per_sample_grad})

        del per_sample_grad

        if self.gradient_clip_val:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        opt.step(closure=closure)
        opt.zero_grad()

        # sch = self.lr_schedulers()
        # sch.step()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_start(self):
        self.dm.sampler.reset()

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

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
    parser.add_argument(
        "-bt", "--balance", choices=["rr", "mean", "pair"], default="mean"
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        default=None,
    )

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

    model_name = f"sam-{args.model.model_name}-da_{args.data.data_augmentation}-{args.model.norm}-backpack-{args.balance}"
    # task_name = args.model.task_name

    L.seed_everything(args.seed)

    dm = CIFAR10DataModule(**vars(args.data))
    dm.prepare_data()
    dm.setup()
    model = BackpackModel(dm=dm, **vars(args.model))

    n = len(dm.train_dataset)
    d = sum(p.numel() for p in model.parameters() if p.requires_grad)

    sampler = SAMSampler(n, d)
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
        ckpt_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
