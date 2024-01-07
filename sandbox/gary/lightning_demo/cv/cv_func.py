import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.cli import LightningCLI

from torchmetrics.functional import accuracy
from functools import partial
from torch.func import functional_call, grad_and_value, vmap

from torchmetrics import Accuracy
import torchopt

from cd2root import cd2root

cd2root()


class CIFAR10DataModule(L.LightningDataModule):
    cifar10_train: Dataset
    cifar10_val: Dataset

    def __init__(
        self,
        data_dir: str = "data/external",
        batch_size: int = 128,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit":
            self.cifar10_train = CIFAR10(
                self.hparams.data_dir, train=True, transform=self.transform
            )
            self.cifar10_val = CIFAR10(
                self.hparams.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=False,
        )


def compute_loss(model, loss_fn, params, buffers, inputs, targets):
    inputs = inputs.unsqueeze(0)
    targets = targets.unsqueeze(0)

    logits = functional_call(model, (params, buffers), (inputs,))

    return loss_fn(logits, targets), logits


class LitClassifier(L.LightningModule):
    def __init__(self, lr=1e-3, wd=5e-4, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(
            num_classes=10, norm_layer=nn.InstanceNorm3d
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.fparams = dict(self.model.named_parameters())
        self.fbuffers = dict(self.model.named_buffers())
        self.fparams = {k: v for k, v in self.fparams.items() if v.requires_grad}
        for v in self.fparams.values():
            v.requires_grad_(False)
        self.metrics = Accuracy(task="multiclass", num_classes=10)

        self.automatic_optimization = False

        self.ft_grad_loss = vmap(
            grad_and_value(
                partial(compute_loss, self.model, self.loss_fn), has_aux=True
            ),
            in_dims=(None, None, 0, 0),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
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

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    # def test_step(self, batch, batch_idx):
    #     return self.evaluate(batch, "test")

    def func_optimizers(self):
        # if self.opt == "sgd":
        #     optimizer = torchopt.sgd(
        #         lr=multi_step_lr(
        #             learning_rate=self.learning_rate,
        #             milestones=[60, 120, 160],
        #             gamma=0.2,
        #         ),
        #         momentum=self.momentum,
        #         weight_decay=self.weight_decay,
        #         nesterov=True,
        #     )
        # elif self.opt == "adam":
        #     no_decay = ["bias", "LayerNorm.weight"]
        #
        #     # mask = torchopt.transform.masked(
        #     #     torchopt.transform.add_decayed_weights(self.weight_decay),
        #     #     lambda name: all(nd not in name for nd in no_decay),
        #     # )
        #     optimizer = torchopt.adamw(
        #         lr=multi_step_lr(
        #             learning_rate=self.learning_rate,
        #             milestones=[60, 120, 160],
        #             gamma=0.2,
        #         ),
        #         betas=(self.adam_beta1, self.adam_beta2),
        #         weight_decay=self.weight_decay,
        #         mask=lambda name: all(nd not in name for nd in no_decay),
        #     )
        # else:
        #     raise NotImplementedError
        optimizer = torchopt.sgd(
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.wd,
            nesterov=True,
        )

        optimizer = torchopt.chain(torchopt.clip_grad_norm(5.0), optimizer)

        return optimizer

    def configure_optimizers(self):
        # torchopt will be passed to correct device
        self.optimizer = self.func_optimizers()
        self.opt_state = self.optimizer.init(self.fparams)

        return torch.optim.SGD(self.parameters(), lr=0)


def multi_step_lr(
    learning_rate: float,
    milestones: list[int],
    gamma: float = 0.1,
):
    def _multi_step_lr(step: int):
        return learning_rate * gamma ** sum(step > m for m in milestones)

    return _multi_step_lr


def main():
    dm = CIFAR10DataModule()
    model = LitClassifier()

    trainer = L.Trainer(
        max_epochs=10,
        precision=32,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
            L.pytorch.callbacks.Timer(),
        ],
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(ckpt_path="best")


if __name__ == "__main__":
    main()
