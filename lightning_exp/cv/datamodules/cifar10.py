import lightning as L
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

import os

from .cv_datamodule import CVDataModule


class CIFAR10DataModule(CVDataModule):
    train_dataset: Dataset
    test_dataset: Dataset

    def __init__(
        self,
        data_dir: str = "data/external",
        train_batch_size: int = 16,
        eval_batch_size: int = 64,
        num_workers: int = os.cpu_count(),
        shuffle: bool = True,
        sampler: Sampler = None,
        **kwargs,
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
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = CIFAR10(
            self.hparams.data_dir, train=True, transform=self.transform
        )
        self.test_dataset = CIFAR10(
            self.hparams.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
