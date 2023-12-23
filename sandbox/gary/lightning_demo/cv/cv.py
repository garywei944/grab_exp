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


class LitClassifier(L.LightningModule):
    def __init__(self, lr=1e-3, wd=5e-4, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(num_classes=10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

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

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.wd,
        )


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
