from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .cv_datamodule import CVDataModule


class CIFAR10DataModule(CVDataModule):
    train_dataset: Dataset
    test_dataset: Dataset

    def __init__(
        self,
        data_dir: str = "data/external",
        train_batch_size: int = 256,
        eval_batch_size: int = 256,
        num_workers: int = 1,
        shuffle: bool = True,
        sampler: Sampler = None,
        data_augmentation: str = "basic",
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.sampler = sampler

        self.save_hyperparameters()

        # TODO: Add data augmentation

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215841, 0.44653091],
                    std=[0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )

        if data_augmentation == "basic":
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    self.transform,
                ]
            )
        elif data_augmentation == "none":
            self.train_transform = self.transform
        else:
            raise NotImplementedError

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = CIFAR10(
            self.data_dir, train=True, transform=self.train_transform
        )
        self.test_dataset = CIFAR10(
            self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            # pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
        )
